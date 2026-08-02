"""The operator pool = the GPT's vocabulary.

In the original GQE paper the pool was chemistry gates (UCCSD excitations).
Those mean nothing for RNA, so we swap in a QAOA-style pool:

  cost tokens  : exp(-i * gamma * H)   -> writes the RNA energy into phases
  mixer tokens : exp(-i * beta * sum X) -> turns those phases into real
                                           differences in probability

Important: you need BOTH. Cost gates alone only change phases, so measuring
would give a flat, uniform distribution. Notebook 03 shows this.
"""

import numpy as np

# Token 0 is reserved as the "start of sequence" marker for the GPT.
# So pool index k is GPT token k + 1.


def make_pool(gammas=None, betas=None):
    """Build the token pool. Returns a dict of arrays plus a size."""
    if gammas is None:
        gammas = [0.15, 0.3, 0.6, 1.2, 2.4, 4.8]
    if betas is None:
        betas = [np.pi / 16, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 3 * np.pi / 4]

    kind = np.array([0] * len(gammas) + [1] * len(betas), dtype=np.int64)
    angle = np.array(list(gammas) + list(betas), dtype=np.float32)
    labels = ([f"cost(g={g:.2f})" for g in gammas] +
              [f"mix(b={b:.2f})" for b in betas])
    return {"kind": kind, "angle": angle, "labels": labels, "size": len(kind)}


def random_token_sequences(pool_size, n_seq, seq_len, rng=None, alternate=False):
    """Random sequences of pool indices, used to build the offline dataset.

    alternate=True forces cost/mixer/cost/mixer, which is plain QAOA.
    Leave it False so the model can discover its own pattern.
    """
    rng = np.random.default_rng(rng)
    return rng.integers(0, pool_size, size=(n_seq, seq_len))


def to_gpt_tokens(pool_indices):
    """Add the start token and shift pool indices by 1."""
    n_seq = pool_indices.shape[0]
    start = np.zeros((n_seq, 1), dtype=np.int64)
    return np.concatenate([start, pool_indices + 1], axis=1)


def from_gpt_tokens(tokens):
    """Strip the start token and shift back to pool indices."""
    return tokens[:, 1:] - 1
