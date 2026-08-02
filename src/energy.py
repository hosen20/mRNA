"""Build the energy of every possible bitstring, fast.

The RNA Hamiltonian is diagonal: each bitstring has one number attached to it.
So we can store all 2^n energies in a single array and reuse it forever.

The trick below writes into numpy *views*, so we never build big temporary
arrays. This is the difference between "runs in 2 seconds" and "runs out of RAM".
"""

import numpy as np


def full_energy_vector(h, J, dtype=np.float32):
    """Energy of all 2^n bitstrings, as one array of length 2^n.

    Bit k of the index is the value of qubit k (little-endian).
    """
    n = len(h)
    size = 1 << n
    E = np.zeros(size, dtype=dtype)

    # Linear terms: add h[k] wherever bit k is 1.
    for k in range(n):
        # View shape: (blocks, 2, stride). The middle axis is bit k.
        E.reshape(-1, 2, 1 << k)[:, 1, :] += h[k]

    # Quadratic terms: add J[a,b] wherever bits a and b are both 1.
    for a in range(n):
        for b in range(a + 1, n):
            if J[a, b] == 0.0:
                continue
            # View shape: (blocks, 2, mid, 2, low) with axis 1 = bit b, axis 3 = bit a.
            view = E.reshape(-1, 2, 1 << (b - a - 1), 2, 1 << a)
            view[:, 1, :, 1, :] += J[a, b]

    return E


def index_to_bits(idx, n):
    """Turn integer state indices into a 0/1 array of shape (..., n)."""
    idx = np.asarray(idx, dtype=np.int64)
    shifts = np.arange(n, dtype=np.int64)
    return ((idx[..., None] >> shifts) & 1).astype(np.int8)


def memory_estimate(n, dtype_bytes=8):
    """Rough RAM for one statevector, in MB. Handy for the scaling section."""
    return (1 << n) * dtype_bytes / 1e6
