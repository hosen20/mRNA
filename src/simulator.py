"""A small, fast statevector simulator built for exactly one job.

Our circuits only ever use two kinds of gate:

  1. cost phase   exp(-i * gamma * H)   with H diagonal -> just multiply phases
  2. mixer        exp(-i * beta * sum_k X_k)  -> a 2x2 rotation on each qubit

Three tricks make this fit in a free Colab session:

  * Batching. We run many circuits at once so numpy/torch work is shared.
  * A phase table. The pool has only a few gamma values, so we precompute
    exp(-i*gamma*E) once and reuse it forever.
  * A cheap readout. We only ever look at the K lowest-energy basis states,
    so reading the energy costs almost nothing.

Backend: uses the GPU through torch when one is available, else numpy.
A general SDK like PennyLane can do the same job but pays Python overhead
per gate. Notebook 03 checks our numbers against PennyLane.
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    torch = None
    HAVE_TORCH = False


def pick_backend(prefer_gpu=True):
    """Return 'torch' if a GPU is available, else 'numpy'."""
    if prefer_gpu and HAVE_TORCH and torch.cuda.is_available():
        return "torch"
    return "numpy"


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------

def scale_energies(E):
    """Rescale to about [-1, 1] so the gamma angles are meaningful."""
    span = float(np.abs(E).max())
    if span == 0.0:
        span = 1.0
    return (E / span).astype(np.float32), span


def low_energy_window(E, K=4096):
    """Indices of the K lowest-energy basis states, sorted best first.

    We only ever care about the good tail, so this is all we need to read out.
    """
    K = min(K, len(E))
    idx = np.argpartition(E, K - 1)[:K]
    idx = idx[np.argsort(E[idx])]
    return idx.astype(np.int64), E[idx].astype(np.float32)


class Sim:
    """Runs batches of circuits. Build once per RNA sequence, reuse forever."""

    def __init__(self, E_scaled, E_raw, pool, K=4096, backend=None,
                 batch=16, dtype_complex=np.complex64):
        self.n = int(round(np.log2(len(E_scaled))))
        self.pool = pool
        self.backend = backend or pick_backend()
        self.batch = batch
        self.low_idx, self.E_low = low_energy_window(E_raw, K)

        # Phase table: one row per cost token, None for mixer tokens.
        self.phase = []
        for k in range(pool["size"]):
            if pool["kind"][k] == 0:
                g = float(pool["angle"][k])
                self.phase.append(np.exp(-1j * g * E_scaled).astype(dtype_complex))
            else:
                self.phase.append(None)

        if self.backend == "torch":
            self.dev = torch.device("cuda")
            self.phase = [None if p is None
                          else torch.as_tensor(p, device=self.dev) for p in self.phase]
            self.low_idx_t = torch.as_tensor(self.low_idx, device=self.dev)
            self.E_low_t = torch.as_tensor(self.E_low, device=self.dev)

    # ---------------- state handling ----------------

    def _fresh_state(self, B):
        size = 1 << self.n
        amp = 1.0 / np.sqrt(size)
        if self.backend == "torch":
            return torch.full((B, size), amp, dtype=torch.complex64, device=self.dev)
        return np.full((B, size), amp, dtype=np.complex64)

    def _apply_cost(self, psi, rows, tk):
        psi[rows] *= self.phase[tk][None, :]

    def _apply_mixer(self, psi, rows, beta):
        """exp(-i*beta*sum X) = the 2x2 matrix [[cos,-i sin],[-i sin,cos]] per qubit."""
        c = complex(np.cos(beta), 0.0)
        s = complex(0.0, -np.sin(beta))
        sub = psi[rows]                       # gather -> a separate buffer
        R = sub.shape[0]
        for k in range(self.n):
            v = sub.reshape(R, -1, 2, 1 << k)
            lo = v[:, :, 0, :]
            hi = v[:, :, 1, :]
            new_lo = c * lo + s * hi
            new_hi = s * lo + c * hi
            v[:, :, 0, :] = new_lo
            v[:, :, 1, :] = new_hi
            sub = v.reshape(R, -1)
        psi[rows] = sub

    def _step(self, psi, tokens):
        """Apply one token per circuit. tokens is a numpy array of shape (B,)."""
        for tk in np.unique(tokens):
            rows = np.nonzero(tokens == tk)[0]
            if self.backend == "torch":
                rows = torch.as_tensor(rows, device=self.dev)
            if self.pool["kind"][tk] == 0:
                self._apply_cost(psi, rows, int(tk))
            else:
                self._apply_mixer(psi, rows, float(self.pool["angle"][tk]))

    # ---------------- readout ----------------

    def _read(self, psi, alpha, shot_floor):
        """Exact CVaR over the low-energy window, plus a realistic 'best'.

        cvar : mean energy of the best alpha fraction of the distribution.
        best : lowest-energy state you would actually see, i.e. one whose
               probability is at least shot_floor.
        """
        if self.backend == "torch":
            amp = psi[:, self.low_idx_t]
            p = (amp.real ** 2 + amp.imag ** 2)
            cum = torch.cumsum(p, dim=1)
            w = torch.clamp(torch.minimum(p, torch.clamp(alpha - cum + p, min=0.0)), min=0.0)
            tot = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
            cvar = (w * self.E_low_t[None, :]).sum(dim=1) / tot.squeeze(1)
            seen = p >= shot_floor
            first = torch.where(seen.any(dim=1),
                                seen.float().argmax(dim=1),
                                torch.zeros(p.shape[0], dtype=torch.long, device=self.dev))
            best = self.E_low_t[first]
            return cvar.cpu().numpy(), best.cpu().numpy(), first.cpu().numpy()

        amp = psi[:, self.low_idx]
        p = (amp.real ** 2 + amp.imag ** 2)
        cum = np.cumsum(p, axis=1)
        w = np.clip(np.minimum(p, np.clip(alpha - cum + p, 0.0, None)), 0.0, None)
        tot = np.maximum(w.sum(axis=1), 1e-12)
        cvar = (w * self.E_low[None, :]).sum(axis=1) / tot
        seen = p >= shot_floor
        first = np.where(seen.any(axis=1), seen.argmax(axis=1), 0)
        best = self.E_low[first]
        return cvar, best, first

    # ---------------- main entry point ----------------

    def run(self, token_batch, alpha=0.15, shots=1024, record_prefix=True):
        """Run a batch of circuits.

        token_batch : (B, N) pool indices.
        Returns cvar and best arrays of shape (B, N) (or (B, 1) if
        record_prefix is False), plus the best basis state index found.
        """
        token_batch = np.asarray(token_batch)
        B, N = token_batch.shape
        psi = self._fresh_state(B)
        floor = 1.0 / float(shots)

        cvars, bests = [], []
        best_val, best_pos = np.inf, 0
        for step in range(N):
            self._step(psi, token_batch[:, step])
            if record_prefix or step == N - 1:
                c, b, first = self._read(psi, alpha, floor)
                cvars.append(c)
                bests.append(b)
                r = int(np.argmin(b))
                if b[r] < best_val:
                    best_val = float(b[r])
                    best_pos = int(first[r])
        return {"cvar": np.stack(cvars, axis=1),
                "best": np.stack(bests, axis=1),
                "best_index": int(self.low_idx[best_pos]),
                "best_energy": best_val}

    def sample_final(self, token_batch, shots=2048, seed=0):
        """Honest shot-based sampling, the way a real QPU would report.

        Slower than run(), so we use it once at the end rather than in the
        training loop.
        """
        rng = np.random.default_rng(seed)
        token_batch = np.asarray(token_batch)
        B, N = token_batch.shape
        psi = self._fresh_state(B)
        for step in range(N):
            self._step(psi, token_batch[:, step])
        if self.backend == "torch":
            p = (psi.real ** 2 + psi.imag ** 2).cpu().numpy()
        else:
            p = (psi.real ** 2 + psi.imag ** 2)
        p = p.astype(np.float64)
        p /= p.sum(axis=1, keepdims=True)
        out = np.empty((B, shots), dtype=np.int64)
        for r in range(B):
            cdf = np.cumsum(p[r])
            cdf[-1] = 1.0
            out[r] = np.searchsorted(cdf, rng.random(shots))
        return out


def to_numpy(x):
    """Get a numpy array back, whether the backend was numpy or torch-on-GPU."""
    if HAVE_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def probs_of(psi):
    """Normalised |psi|^2 as numpy, from either backend."""
    a = to_numpy(psi)
    p = (a.real ** 2 + a.imag ** 2).astype(np.float64)
    return p / p.sum(axis=-1, keepdims=True)
