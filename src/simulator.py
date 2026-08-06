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

def scale_energies(E, headroom=1.0):
    """Rescale energies so the gamma angles actually do something.

    Two traps here, and we hit both.

    Trap 1: dividing by max|E|. Clash penalties push the worst bitstrings to
    enormous energies, so the real structural differences get squashed to
    nothing and the cost gate cannot tell good structures apart.

    Trap 2: clipping at a quantile. At 27 qubits the 0.01% quantile is still
    +92, because only 115 of 134 million bitstrings are valid structures. The
    quantile scales with 2^n, so it drifts away from the region we care about
    as the problem grows.

    Fix: clip at |E_min|, which tracks the good region at any size. The useful
    energies then sit in [-1, 1] and gamma*dE lands near 1 radian, where
    interference is strong. We do not care how bad a bad structure is, only
    that it is bad, so flattening the tail costs us nothing.
    """
    lo = float(np.min(E))
    hi = headroom * abs(lo) if lo < 0 else float(np.max(E))
    if hi <= 0:
        hi = 1.0
    Ec = np.minimum(E, hi)
    span = max(abs(lo), abs(hi)) or 1.0
    return (Ec / span).astype(np.float32), span


def suggest_alpha(E, floor=5e-4, cap=0.1):
    """Pick the CVaR fraction to match how rare good structures are.

    CVaR averages the best alpha fraction of the distribution. If alpha is far
    bigger than the fraction of bitstrings that are valid structures, that
    average is dominated by garbage and the training signal nearly vanishes.
    """
    good = float((E < 0).mean())
    return float(np.clip(good if good > 0 else floor, floor, cap))


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
                 batch=16, dtype_complex=np.complex64, cache_phases=None,
                 init_probs=None, shots=1024):
        self.n = int(round(np.log2(len(E_scaled))))
        self.pool = pool
        self.backend = backend or pick_backend()
        self.batch = batch
        self.low_idx, self.E_low = low_energy_window(E_raw, K)
        self.alpha = suggest_alpha(E_raw)
        self.E_opt = float(np.min(E_raw))   # true optimum, for opt_rate
        # One shot budget for the whole object, so training and the final
        # evaluation are measured the same way. Mixing them makes 'best'
        # look worse than CVaR, which is impossible and very confusing.
        self.shots = shots
        if 1.0 / shots > self.alpha:
            print(f'  !! shots={shots} is too few for alpha={self.alpha:.2e}.'
                  f' best will look worse than CVaR.'
                  f' Use at least {int(np.ceil(2/self.alpha))} shots.')

        # Caching exp(-i*gamma*E) for every gamma is fast but costs
        # (number of gammas) x 2^n x 8 bytes. At 27 qubits that is 1.07 GB per
        # gamma, which will not fit on a free T4 alongside the statevectors.
        # Above 24 qubits we therefore recompute the phase each time from a
        # single stored copy of E. Recomputing costs a few ms on GPU.
        if cache_phases is None:
            cache_phases = self.n <= 24
        self.cache_phases = cache_phases

        if self.backend == "torch":
            self.dev = torch.device("cuda")
            self.E_dev = torch.as_tensor(np.asarray(E_scaled, dtype=np.float32),
                                         device=self.dev)
            self.low_idx_t = torch.as_tensor(self.low_idx, device=self.dev)
            self.E_low_t = torch.as_tensor(self.E_low, device=self.dev)
        else:
            self.E_dev = np.asarray(E_scaled, dtype=np.float32)

        self._init_vec = None
        if init_probs is not None:
            self.set_warm_start(init_probs)

        self.phase = []
        for k in range(pool["size"]):
            if pool["kind"][k] == 0 and cache_phases:
                g = float(pool["angle"][k])
                ph = np.exp(-1j * g * E_scaled).astype(dtype_complex)
                if self.backend == "torch":
                    ph = torch.as_tensor(ph, device=self.dev)
                self.phase.append(ph)
            else:
                self.phase.append(None)

    def set_warm_start(self, probs):
        """Start from a biased product state instead of |+>^n.

        probs[k] is the chance qubit k starts as 1. This is the RNA analogue of
        the Hartree-Fock reference state in the original GQE paper: a cheap
        classical guess that the quantum circuit then improves on.

        We only use each stem's own energy h[k], which says nothing about which
        stems fit together, so the answer is emphatically not being handed over.
        This is warm-start QAOA (Egger et al. 2021). Say so in your write-up.
        """
        probs = np.clip(np.asarray(probs, dtype=np.float64), 0.02, 0.98)
        vec = np.array([1.0], dtype=np.complex64)
        for k in range(self.n):
            q = np.array([np.sqrt(1 - probs[k]), np.sqrt(probs[k])], dtype=np.complex64)
            vec = np.kron(q, vec)          # little-endian: qubit 0 fastest
        vec /= np.linalg.norm(vec)
        if self.backend == "torch":
            self._init_vec = torch.as_tensor(vec, device=self.dev)
        else:
            self._init_vec = vec

    def _phase_for(self, tk):
        """exp(-i*gamma*E), from cache or computed on the spot."""
        if self.phase[tk] is not None:
            return self.phase[tk]
        g = float(self.pool["angle"][tk])
        if self.backend == "torch":
            ang = self.E_dev * (-g)
            return torch.polar(torch.ones_like(ang), ang)
        return np.exp(-1j * g * self.E_dev).astype(np.complex64)

    def memory_gb(self, batch=None):
        """Rough GPU memory this configuration needs."""
        b = batch or self.batch
        cached = sum(1 for p in self.phase if p is not None)
        vecs = b + cached + (0 if self.cache_phases else 1.5)   # +E, +one temp
        return vecs * (1 << self.n) * 8 / 1e9

    # ---------------- state handling ----------------

    def _fresh_state(self, B):
        size = 1 << self.n
        if self._init_vec is None:
            amp = 1.0 / np.sqrt(size)
            if self.backend == "torch":
                return torch.full((B, size), amp, dtype=torch.complex64, device=self.dev)
            return np.full((B, size), amp, dtype=np.complex64)
        if self.backend == "torch":
            return self._init_vec.unsqueeze(0).repeat(B, 1).clone()
        return np.repeat(self._init_vec[None, :], B, axis=0).copy()

    def _apply_cost(self, psi, rows, tk):
        """Multiply the given rows by exp(-i*gamma*E). No large temporary."""
        if len(rows) == 0:
            return
        psi[rows] *= self._phase_for(tk)[None, :]

    def _apply_mixer(self, psi, rows, beta, wires=None):
        """exp(-i*beta*sum X) = the 2x2 matrix [[cos,-i sin],[-i sin,cos]] per qubit."""
        c = complex(np.cos(beta), 0.0)
        s = complex(0.0, -np.sin(beta))
        # Fast path: if every circuit in the batch uses this token we can work
        # on psi in place. At 27 qubits the gather alone moves gigabytes.
        full = len(rows) == psi.shape[0]
        sub = psi if full else psi[rows]
        R = sub.shape[0]
        qubits = range(self.n) if wires is None else wires
        for k in qubits:
            k = int(k)
            v = sub.reshape(R, -1, 2, 1 << k)
            lo = v[:, :, 0, :]
            hi = v[:, :, 1, :]
            new_lo = c * lo + s * hi
            new_hi = s * lo + c * hi
            v[:, :, 0, :] = new_lo
            v[:, :, 1, :] = new_hi
            sub = v.reshape(R, -1)
        if not full:
            psi[rows] = sub

    def _step(self, psi, tokens):
        """Apply one token per circuit. tokens is a numpy array of shape (B,)."""
        for tk in np.unique(tokens):
            rows = np.nonzero(tokens == tk)[0]
            if self.backend == "torch":
                rows = torch.as_tensor(rows, device=self.dev)
            if self.pool["kind"][tk] == 0:
                self._apply_cost(psi, rows, int(tk))  # noqa
            else:
                w = self.pool.get("wires", [None] * self.pool["size"])[tk]
                self._apply_mixer(psi, rows, float(self.pool["angle"][tk]), wires=w)

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
            # "best" = the best energy you could actually expect to observe.
            # Use CUMULATIVE probability, so it reflects the real mass sitting
            # on good states, and fall back to the WORST entry when the
            # distribution is too diffuse to see anything.
            enough = cum >= shot_floor
            last = torch.full((p.shape[0],), p.shape[1] - 1,
                              dtype=torch.long, device=self.dev)
            first = torch.where(enough.any(dim=1), enough.float().argmax(dim=1), last)
            best = self.E_low_t[first]
            return cvar.cpu().numpy(), best.cpu().numpy(), first.cpu().numpy()

        amp = psi[:, self.low_idx]
        p = (amp.real ** 2 + amp.imag ** 2)
        cum = np.cumsum(p, axis=1)
        w = np.clip(np.minimum(p, np.clip(alpha - cum + p, 0.0, None)), 0.0, None)
        tot = np.maximum(w.sum(axis=1), 1e-12)
        cvar = (w * self.E_low[None, :]).sum(axis=1) / tot
        enough = cum >= shot_floor
        first = np.where(enough.any(axis=1), enough.argmax(axis=1), p.shape[1] - 1)
        best = self.E_low[first]
        return cvar, best, first

    # ---------------- main entry point ----------------

    def run(self, token_batch, alpha=None, shots=None, record_prefix=True):
        """Run a batch of circuits.

        token_batch : (B, N) pool indices.
        Returns cvar and best arrays of shape (B, N) (or (B, 1) if
        record_prefix is False), plus the best basis state index found.
        """
        token_batch = np.asarray(token_batch)
        if alpha is None:
            alpha = self.alpha
        if shots is None:
            shots = self.shots
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
                    best_row = r
        return {"cvar": np.stack(cvars, axis=1),
                "best": np.stack(bests, axis=1),
                "best_index": int(self.low_idx[best_pos]),
                "best_energy": best_val,
                "best_row": int(best_row)}

    def sample_final(self, token_batch, shots=None, seed=0):
        """Honest shot-based sampling, the way a real QPU would report.

        Slower than run(), so we use it once at the end rather than in the
        training loop.
        """
        rng = np.random.default_rng(seed)
        if shots is None:
            shots = self.shots
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
