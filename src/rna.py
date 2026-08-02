"""RNA side of the problem: stems, QUBO, and decoding back to dot-bracket.

The whole idea in one line:
    RNA structure  ->  pick a set of stems  ->  one qubit per stem  ->  QUBO.
"""

import numpy as np

# Base pairs we allow. AU/GC are Watson-Crick, GU is the "wobble" pair.
PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}

# A hairpin loop needs at least 3 free bases, so j - i - 1 >= 3.
MIN_LOOP = 3

# Try to load ViennaRNA. If it is missing we fall back to a rough model.
try:
    import RNA as _vienna
    HAVE_VIENNA = True
except ImportError:
    _vienna = None
    HAVE_VIENNA = False


# --------------------------------------------------------------------------
# 1. Sequences
# --------------------------------------------------------------------------

def random_sequence(length, rng=None, gc=0.5):
    """Make a random RNA sequence. gc = fraction of G/C bases."""
    rng = np.random.default_rng(rng)
    p = [(1 - gc) / 2, (1 - gc) / 2, gc / 2, gc / 2]  # A, U, G, C
    return "".join(rng.choice(list("AUGC"), size=length, p=p))


def can_pair(a, b):
    return (a, b) in PAIRS


# --------------------------------------------------------------------------
# 2. Stems
# --------------------------------------------------------------------------

def find_stems(seq, min_len=3):
    """Find all maximal stems of at least `min_len` base pairs.

    A stem is a run of pairs (i,j), (i+1,j-1), (i+2,j-2), ...
    "Maximal" means we cannot make it longer at either end.
    Returns a list of dicts with keys: i, j, length, pairs.
    """
    L = len(seq)
    stems = []
    for i in range(L):
        for j in range(i + MIN_LOOP + 1, L):
            if not can_pair(seq[i], seq[j]):
                continue
            # Skip if this pair is in the middle of a longer stem.
            if i > 0 and j + 1 < L and can_pair(seq[i - 1], seq[j + 1]):
                continue
            # Walk inward as far as we can.
            k = 0
            while (j - k) - (i + k) - 1 >= MIN_LOOP and can_pair(seq[i + k], seq[j - k]):
                k += 1
            if k >= min_len:
                pairs = [(i + t, j - t) for t in range(k)]
                stems.append({"i": i, "j": j, "length": k, "pairs": pairs})
    # Longest / most promising stems first. Keeps qubit order stable.
    stems.sort(key=lambda s: (-s["length"], s["i"], s["j"]))
    return stems


def stems_to_dotbracket(seq_len, stems):
    """Turn a list of stems into a dot-bracket string."""
    db = ["."] * seq_len
    for s in stems:
        for (a, b) in s["pairs"]:
            db[a] = "("
            db[b] = ")"
    return "".join(db)


def stems_overlap(s, t):
    """True if two stems use the same base (a base can pair only once)."""
    bases_s = set()
    for (a, b) in s["pairs"]:
        bases_s.add(a)
        bases_s.add(b)
    for (a, b) in t["pairs"]:
        if a in bases_s or b in bases_s:
            return True
    return False


def stems_cross(s, t):
    """True if two stems form a pseudoknot (their pairs cross).

    Pairs (a,b) and (c,d) cross when a < c < b < d.
    """
    for (a, b) in s["pairs"]:
        for (c, d) in t["pairs"]:
            if a < c < b < d or c < a < d < b:
                return True
    return False


# --------------------------------------------------------------------------
# 3. Energies from ViennaRNA
# --------------------------------------------------------------------------

def set_vienna_defaults(temperature=37.0, dangles=2, no_lonely_pairs=False):
    """Fix the ViennaRNA settings so our numbers are reproducible."""
    if not HAVE_VIENNA:
        return
    _vienna.cvar.temperature = temperature
    _vienna.cvar.dangles = dangles
    _vienna.cvar.noLonelyPairs = 1 if no_lonely_pairs else 0


def eval_structure(seq, db):
    """Free energy (kcal/mol) of a dot-bracket structure. Uses ViennaRNA."""
    if HAVE_VIENNA:
        return float(_vienna.fold_compound(seq).eval_structure(db))
    return _fallback_energy(seq, db)


def mfe_structure(seq):
    """The reference answer: ViennaRNA's minimum free energy structure."""
    if HAVE_VIENNA:
        db, e = _vienna.fold(seq)
        return db, float(e)
    raise RuntimeError("ViennaRNA is needed for the reference MFE. "
                       "Run: pip install ViennaRNA")


def _fallback_energy(seq, db):
    """Crude backup energy model, only used if ViennaRNA is missing.

    -2.0 per stacked pair, +4.0 per hairpin. Good enough to smoke-test code,
    NOT good enough for real benchmarks.
    """
    pairs = dotbracket_to_pairs(db)
    pset = set(pairs)
    stacks = sum(1 for (a, b) in pairs if (a + 1, b - 1) in pset)
    hairpins = sum(1 for (a, b) in pairs if (a + 1, b - 1) not in pset)
    return -2.0 * stacks + 4.0 * hairpins


# --------------------------------------------------------------------------
# 4. Building the QUBO
# --------------------------------------------------------------------------

def build_qubo(seq, stems, penalty=None, allow_pseudoknots=False,
               two_body=True):
    """Build the QUBO for one RNA sequence.

    Energy of a bitstring x is:   sum_s h[s]*x[s]  +  sum_{s<t} J[s,t]*x[s]*x[t]

    h[s]   = free energy of stem s on its own.
    J[s,t] = a big positive penalty if the two stems cannot coexist, otherwise
             the real interaction energy between them.

    About that interaction term (this matters a lot). Stem energies do NOT
    simply add up. ViennaRNA charges a hairpin loop penalty of roughly
    +4 kcal/mol to a lone stem. Put two nested stems together and that penalty
    is paid once, not twice, because the outer hairpin becomes an interior
    loop. If we set J = 0 for compatible stems, the model thinks adding a
    second stem costs +4 when it actually gains about -4, and it stops after
    one stem.

    So we measure the interaction instead of assuming it is zero:

        J[s,t] = E({s,t}) - h[s] - h[t]

    This is a two-body cluster expansion. It is **exact** for any pair of
    stems, and a good approximation for three or more. Set two_body=False to
    see how bad the naive additive version is.

    Returns (h, J) as numpy arrays. J is upper-triangular.
    """
    n = len(stems)
    L = len(seq)

    h = np.zeros(n, dtype=np.float64)
    for k, s in enumerate(stems):
        h[k] = eval_structure(seq, stems_to_dotbracket(L, [s]))

    # Which pairs of stems clash?
    clash = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(a + 1, n):
            bad = stems_overlap(stems[a], stems[b])
            if not bad and not allow_pseudoknots:
                bad = stems_cross(stems[a], stems[b])
            clash[a, b] = bad

    # Real interaction energy for the pairs that can coexist.
    J = np.zeros((n, n), dtype=np.float64)
    if two_body:
        for a in range(n):
            for b in range(a + 1, n):
                if clash[a, b]:
                    continue
                e_ab = eval_structure(seq, stems_to_dotbracket(L, [stems[a], stems[b]]))
                J[a, b] = e_ab - h[a] - h[b]

    # The clash penalty must outweigh every real energy in the problem.
    if penalty is None:
        biggest = max(float(np.abs(h).max()) if n else 1.0,
                      float(np.abs(J).max()) if n else 1.0)
        penalty = max(10.0, 3.0 * biggest)
    J[clash] = penalty

    return h, J


def qubo_energy(h, J, bits):
    """Energy of one or many bitstrings. bits has shape (n,) or (B, n)."""
    bits = np.atleast_2d(np.asarray(bits, dtype=np.float64))
    lin = bits @ h
    quad = np.einsum("bi,ij,bj->b", bits, J, bits)
    return lin + quad


# --------------------------------------------------------------------------
# 5. Decoding: bitstring -> valid structure
# --------------------------------------------------------------------------

def decode_bits(seq, stems, bits, allow_pseudoknots=False, h=None):
    """Turn a bitstring into a valid dot-bracket structure.

    We take the chosen stems, sort them best-energy-first, and greedily keep
    the ones that do not clash. This "repair" step means every bitstring maps
    to a real structure, so the model never wastes time on invalid answers.

    Pass `h` (the stem energies from build_qubo) to avoid re-calling ViennaRNA.
    """
    bits = np.asarray(bits).astype(int).ravel()
    picked = [k for k in range(len(stems)) if bits[k] == 1]
    if h is None:
        h = np.array([eval_structure(seq, stems_to_dotbracket(len(seq), [stems[k]]))
                      for k in range(len(stems))])
    # Best (most negative) energy first.
    picked.sort(key=lambda k: h[k])
    chosen = [stems[k] for k in picked]

    kept = []
    for s in chosen:
        clash = False
        for t in kept:
            if stems_overlap(s, t):
                clash = True
                break
            if not allow_pseudoknots and stems_cross(s, t):
                clash = True
                break
        if not clash:
            kept.append(s)
    return stems_to_dotbracket(len(seq), kept)


def dotbracket_to_pairs(db):
    """List of (i, j) base pairs from a dot-bracket string."""
    stack, pairs = [], []
    for k, c in enumerate(db):
        if c == "(":
            stack.append(k)
        elif c == ")":
            if stack:
                pairs.append((stack.pop(), k))
    return sorted(pairs)


def limit_stems(stems, h, max_n):
    """Keep only the `max_n` most stabilising stems.

    Use this to dial the qubit count to whatever your machine can simulate.
    Returns (stems_subset, indices_kept).
    """
    if len(stems) <= max_n:
        return stems, np.arange(len(stems))
    keep = np.argsort(h)[:max_n]
    keep = np.sort(keep)
    return [stems[k] for k in keep], keep


def helices_from_dotbracket(db):
    """Split a structure into its helices: list of (i, j, length)."""
    pairs = dict(dotbracket_to_pairs(db))
    helices, used = [], set()
    for (i, j) in sorted(pairs.items()):
        if i in used:
            continue
        k = 0
        while (i + k) in pairs and pairs.get(i + k) == j - k:
            used.add(i + k)
            k += 1
        helices.append((i, j, k))
    return helices


def check_mfe_coverage(seq, stems, ref_db=None, verbose=True):
    """Can our stem set even express the true MFE?

    This catches the most common encoding mistake: the right helix was filtered
    out by min_len, so no optimiser could ever find it. Run this BEFORE blaming
    the optimiser for a bad gap.
    """
    if ref_db is None:
        ref_db, _ = mfe_structure(seq)
    have = {(s["i"], s["j"], s["length"]) for s in stems}
    starts = {(s["i"], s["j"]): s["length"] for s in stems}

    missing = []
    for (i, j, k) in helices_from_dotbracket(ref_db):
        ok = (i, j, k) in have or starts.get((i, j), -1) >= k
        if not ok:
            missing.append((i, j, k))
        if verbose:
            print(f"  helix {i:3d}-{j:3d} length {k}   "
                  f"{'in our stem set' if ok else 'MISSING'}")
    if verbose:
        if missing:
            shortest = min(k for (_, _, k) in missing)
            print(f"\n  {len(missing)} helix/helices missing. Shortest is {shortest} bp.")
            # Was min_len too high, or was the stem found and then trimmed?
            min_len_used = min(t["length"] for t in stems) if stems else 99
            if shortest >= min_len_used:
                print("  -> these exist at this min_len but were trimmed out.")
                print("     Raise max_n in build_encoding.")
            else:
                print(f"  -> lower min_len to {shortest}, or the gap can never reach 0.")
        else:
            print("\n  All MFE helices are representable. Any gap is now the")
            print("  optimiser's or the two-body approximation's, not coverage.")
    return missing


def stacking_energies(seq, stems):
    """Energy of each stem with the hairpin closure cost removed.

    A lone 3 bp stem pays about +4.5 kcal/mol to close its loop, which can make
    its total energy positive even when its stacking is excellent. Ranking by
    total energy therefore throws away good short helices. Ranking by stacking
    does not.
    """
    L = len(seq)
    out = np.zeros(len(stems))
    for k, s in enumerate(stems):
        whole = eval_structure(seq, stems_to_dotbracket(L, [s]))
        inner = s["pairs"][-1]
        lone = {"pairs": [inner]}
        out[k] = whole - eval_structure(seq, stems_to_dotbracket(L, [lone]))
    return out


def build_encoding(seq, min_len=3, max_n=20, allow_pseudoknots=False,
                   verbose=True):
    """One call: stems -> trimmed to max_n qubits -> QUBO.

    Stems are ranked by stacking energy, not total energy, so short but strong
    helices survive the cut.
    """
    stems = find_stems(seq, min_len=min_len)
    if verbose:
        print(f"min_len={min_len}: {len(stems)} stems found")
    if len(stems) > max_n:
        score = stacking_energies(seq, stems)
        keep = np.sort(np.argsort(score)[:max_n])
        stems = [stems[k] for k in keep]
        if verbose:
            print(f"trimmed to the {max_n} with the strongest stacking")
    h, J = build_qubo(seq, stems, allow_pseudoknots=allow_pseudoknots)
    if verbose:
        print(f"-> {len(stems)} qubits")
    return stems, h, J


def rerank(seq, stems, indices, n, h=None, allow_pseudoknots=False,
           verbose=False):
    """Score candidate bitstrings with ViennaRNA and return the best.

    Why this exists. The QUBO is quadratic, so it can only capture pair
    interactions between stems. With three or more stems the real energy has
    higher-order terms that a QUBO simply cannot hold. That is a property of
    QUBOs, not a bug we can patch, and it applies to real hardware too.

    So we split the work the way hybrid algorithms normally do:
      the quantum part narrows 2^n bitstrings down to a few candidates,
      then ViennaRNA scores those candidates exactly.

    Structures repeat a lot after repair, so we only evaluate each one once.

    indices : basis-state indices, e.g. the lowest-QUBO-energy states or the
              states the circuit actually sampled.
    """
    from .energy import index_to_bits

    seen = {}
    for idx in np.atleast_1d(np.asarray(indices, dtype=np.int64)):
        bits = index_to_bits(np.array([idx]), n)[0]
        db = decode_bits(seq, stems, bits, allow_pseudoknots=allow_pseudoknots, h=h)
        if db not in seen:
            seen[db] = eval_structure(seq, db)

    best_db = min(seen, key=seen.get)
    if verbose:
        print(f"  {len(indices)} candidates -> {len(seen)} distinct structures")
        print(f"  best: {seen[best_db]:.2f} kcal/mol")
    return best_db, seen[best_db], seen


def best_from_energy_vector(seq, stems, E, n, top_k=200, h=None,
                            allow_pseudoknots=False, verbose=False):
    """Take the top_k lowest-QUBO-energy states and rerank them exactly."""
    top_k = min(top_k, len(E))
    idx = np.argpartition(E, top_k - 1)[:top_k]
    return rerank(seq, stems, idx, n, h=h,
                  allow_pseudoknots=allow_pseudoknots, verbose=verbose)
