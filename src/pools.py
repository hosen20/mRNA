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


def make_pool(gammas=None, betas=None, n_qubits=None, groups=4,
              group_betas=None):
    """Build the token pool (the GPT's vocabulary).

    Basic pool: global cost phases and a global X mixer.

    The problem with a global mixer is that every qubit gets the SAME angle.
    The circuit therefore cannot say "amplify stem 3, suppress stem 7", which
    is exactly what you need when only ~100 of 4 million bitstrings are valid
    structures. That is what makes training plateau.

    Pass n_qubits to also get GROUP mixers: the stems are split into `groups`
    bands by energy rank, and each token mixes only one band. The model can
    then treat strong and weak helices differently. Tokens are defined by rank,
    not by absolute qubit index, so the same vocabulary works for any sequence.
    """
    if gammas is None:
        gammas = [0.15, 0.3, 0.6, 1.2, 2.4, 4.8]
    if betas is None:
        betas = [np.pi / 16, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 3 * np.pi / 4]
    if group_betas is None:
        group_betas = [np.pi / 8, np.pi / 4, np.pi / 2]

    kind, angle, labels, wires = [], [], [], []

    for g in gammas:
        kind.append(0); angle.append(g); wires.append(None)
        labels.append(f"cost(g={g:.2f})")

    for b in betas:
        kind.append(1); angle.append(b); wires.append(None)
        labels.append(f"mix_all(b={b:.2f})")

    if n_qubits is not None and groups > 1:
        # Qubits are already ordered strongest-stem-first by build_encoding.
        bands = np.array_split(np.arange(n_qubits), groups)
        for gi, band in enumerate(bands):
            if len(band) == 0:
                continue
            for b in group_betas:
                kind.append(1); angle.append(b); wires.append(band.astype(np.int64))
                labels.append(f"mix_g{gi}(b={b:.2f})")

    return {"kind": np.array(kind, dtype=np.int64),
            "angle": np.array(angle, dtype=np.float32),
            "wires": wires, "labels": labels, "size": len(kind)}


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


def warm_start_probs(h, temperature=None):
    """Initial per-qubit probabilities from each stem's own energy.

    Strongly stabilising stems start with a higher chance of being on. Uses
    only h, never J, so it carries no information about which stems are
    compatible - that is the part the circuit has to work out.
    """
    h = np.asarray(h, dtype=np.float64)
    if temperature is None:
        temperature = max(1e-6, float(np.abs(h).mean()))
    return 1.0 / (1.0 + np.exp(h / temperature))
