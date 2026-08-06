"""Memory-lean simulator, used ONLY by notebook 06 (the 27-qubit run).

Nothing else imports this file. `src/simulator.py` is untouched, so notebooks
01 to 05 behave exactly as before.

Why this exists. At 27 qubits one statevector is 1.07 GB. Two things in the
base simulator are fine at 22 qubits but run a free T4 out of memory at 27:

  1. `_fresh_state` did `.repeat(B, 1).clone()`. `.repeat` already returns a
     new tensor, so the `.clone()` allocated a second copy of the whole batch
     and threw the first away. At 8 circuits that is 8.6 GB wasted.

  2. `run()` allocated one statevector per circuit in the request. Callers that
     pass more circuits than `batch` (for example the offline evaluation, which
     uses `n_eval`) would ask for the lot in one go.

Both are memory-only changes. The maths is identical: circuits evolve
independently and `run()` uses no random numbers, so splitting a batch gives
bit-identical results. Notebook 06 checks this at the start.
"""

import numpy as np

from . import simulator as _base
from .simulator import (                      # re-exported for convenience
    scale_energies, suggest_alpha, low_energy_window,
    pick_backend, probs_of, to_numpy, HAVE_TORCH,
)

try:
    import torch
except ImportError:
    torch = None


class Sim(_base.Sim):
    """Same simulator, kinder to GPU memory."""

    def _fresh_state(self, B):
        size = 1 << self.n
        if self._init_vec is None:
            amp = 1.0 / np.sqrt(size)
            if self.backend == "torch":
                return torch.full((B, size), amp, dtype=torch.complex64,
                                  device=self.dev)
            return np.full((B, size), amp, dtype=np.complex64)
        if self.backend == "torch":
            # .repeat already copies. No .clone() here.
            return self._init_vec.unsqueeze(0).repeat(B, 1)
        return np.repeat(self._init_vec[None, :], B, axis=0)

    def _free(self):
        """Hand the statevector memory back before the next allocation."""
        if self.backend == "torch" and torch is not None:
            torch.cuda.empty_cache()

    def run(self, token_batch, alpha=None, shots=None, record_prefix=True):
        """Same as the base run(), but never holds more than `batch` states."""
        token_batch = np.asarray(token_batch)
        B = token_batch.shape[0]

        if B <= self.batch:
            out = super().run(token_batch, alpha=alpha, shots=shots,
                              record_prefix=record_prefix)
            self._free()
            return out

        parts = []
        for i in range(0, B, self.batch):
            parts.append(super().run(token_batch[i:i + self.batch], alpha=alpha,
                                     shots=shots, record_prefix=record_prefix))
            self._free()

        merged = {"cvar": np.concatenate([p["cvar"] for p in parts], axis=0),
                  "best": np.concatenate([p["best"] for p in parts], axis=0)}
        k = int(np.argmin([p["best_energy"] for p in parts]))
        merged["best_energy"] = parts[k]["best_energy"]
        merged["best_index"] = parts[k]["best_index"]
        merged["best_row"] = k * self.batch + parts[k].get("best_row", 0)
        return merged
