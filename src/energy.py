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


def check_energy_vector(h, J, E, n_samples=50000, rng=None, tol=1e-4):
    """Verify the energy vector on a random sample of states.

    Do NOT check all 2^n states above about 20 qubits. Building the full bit
    matrix costs 2^n x n x 8 bytes once it is cast to float: 738 MB at 22
    qubits, but 29 GB at 27. That crashes Colab. A random sample of 50k states
    catches any real bug just as well.
    """
    rng = np.random.default_rng(rng)
    n = len(h)
    idx = rng.integers(0, 1 << n, size=min(n_samples, 1 << n))
    bits = index_to_bits(idx, n).astype(np.float64)
    direct = bits @ h + np.einsum("bi,ij,bj->b", bits, J, bits)
    worst = float(np.abs(np.asarray(E)[idx] - direct).max())
    print(f"checked {len(idx)} random states, max difference: {worst:.2e}")
    if worst <= tol:
        print("-> energy vector is correct")
    else:
        print("-> MISMATCH. Something is wrong with the QUBO or the builder.")
    return worst


def pick_dtype(n, budget_gb=0.5):
    """float64 while it is cheap, float32 once the vector gets big."""
    return np.float64 if (1 << n) * 8 / 1e9 <= budget_gb else np.float32
