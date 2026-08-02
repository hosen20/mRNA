# GQE for RNA secondary structure prediction

WISER 2026 / Moderna challenge: *Optimization of mRNA Secondary Structure
Prediction Using Quantum Computing*.

We adapt the **Generative Quantum Eigensolver** (GQE, Nakaji et al. 2024) from
molecular ground states to RNA folding. A small GPT learns to *write quantum
circuits* whose measurement outcomes are low-energy RNA structures. Results are
checked against ViennaRNA.

---

## Quick start on Google Colab

**Every notebook runs on its own.** You upload one notebook at a time; the code
and the saved results live in your Drive.

**Do this once:**

1. Upload **`gqe-rna.zip`** to the top level of your Google Drive (My Drive).
   Do not unzip it. The notebooks handle that.

**Then for each notebook, in order 01 to 05:**

2. Go to [colab.research.google.com](https://colab.research.google.com) ->
   **File -> Upload notebook** -> pick the notebook from `notebooks/`.
3. **Runtime -> Change runtime type -> T4 GPU** (free tier is fine).
4. **Runtime -> Run all**. Approve the Drive popup when it appears.

Each notebook saves its results into `gqe-rna/results/` **in your Drive**, so
they survive when the runtime shuts down, and the next notebook picks them up
automatically. If a cell reports a missing file, it tells you which notebook to
run first.

> Opening `.ipynb` files straight from Drive shows raw text unless Colab is
> connected to your account. Using **File -> Upload notebook** avoids that
> entirely. To fix it permanently: in Drive, right-click any `.ipynb` ->
> **Open with -> Connect more apps** -> install **Colaboratory**.

**Total runtime on free Colab: about 40 minutes for all five notebooks.**

If you ever see `ModuleNotFoundError: No module named 'src'`, run this in a new
cell. It happens when Python looked at the Drive folder before it existed:

```python
import sys, os, importlib
REPO = "/content/drive/MyDrive/gqe-rna"
if REPO not in sys.path: sys.path.insert(0, REPO)
importlib.invalidate_caches()
from src import colab_utils as cu
cu.setup(repo=REPO); cu.install_vienna()
```

Helper functions live in `src/colab_utils.py`:
`cu.setup()`, `cu.save()`, `cu.require()`, `cu.show_results()`,
`cu.download_results()`, `cu.upload_into_results()`.

---

## What is in here

```
src/
  rna.py         stems, QUBO construction, decode + repair, ViennaRNA helpers
  energy.py      energy of all 2^n bitstrings, built with numpy views
  simulator.py   fast batched statevector simulator (GPU via torch, else numpy)
  pools.py       the token vocabulary the GPT chooses from
  gpt.py         small transformer + logit-matching and GRPO losses
  train.py       offline and online training loops, replay buffer
  baselines.py   brute force, random, greedy, simulated annealing, ablation
  metrics.py     base-pair F1, energy gap, exact match
  colab_utils.py Drive setup, saving results between notebooks, downloads
notebooks/
  01_setup_and_classical.ipynb    ViennaRNA reference answers
  02_encoding_and_qubo.ipynb      RNA -> stems -> qubits -> QUBO
  03_gqe_training.ipynb           the main event: GPT writes circuits
  04_baselines_and_metrics.ipynb  is the quantum part actually helping?
  05_scaling_and_extras.ipynb     qubit scaling, pseudoknots, noise, limits
```

---

## The approach in six lines

1. Enumerate all **stems** (runs of consecutive base pairs) in the sequence.
2. One qubit per stem. Qubit = 1 means "use this stem".
3. Energy = stem energies + big penalties for stems that clash.
4. A GPT picks a sequence of gates from a menu of `exp(-i*gamma*H)` (cost) and
   `exp(-i*beta*sum X)` (mixer) operations.
5. Measure the circuit, decode the bitstring into a structure, score it.
6. Train the GPT so the circuits it writes produce lower-energy structures.

---

## Three changes we had to make to GQE

The original algorithm targets molecules. RNA folding is different in ways that
matter:

**1. The objective. (most important)**
In chemistry the Hamiltonian is non-diagonal and the ground state is a
superposition, so minimising the average energy is correct. Our Hamiltonian is
diagonal: every answer is a basis state, and the average energy is the average
over *all* structures, which is not what we want. We minimise **CVaR** instead:
the mean energy of the best 15% of the distribution. Get this wrong and the
model optimises the wrong thing while the loss curve still looks fine.

**2. The operator pool.**
UCCSD excitation gates mean nothing for RNA. We use a QAOA-style pool of cost
and mixer tokens. Cost gates alone only change phases and produce a flat
measurement distribution; notebook 03 demonstrates this.

**3. A decode and repair step.**
Chemistry has no equivalent. A raw bitstring may pick clashing stems, so we
greedily keep the best non-clashing ones. Every bitstring therefore maps to a
valid structure and the model never wastes effort on invalid answers.

Smaller changes: initial state is `|+>^n` instead of Hartree-Fock; the model is
1.5M parameters instead of 85M; the training loop adds an online phase with a
replay buffer, which the PennyLane demo does not have.

---

## Performance notes (why this runs on free Colab)

The PennyLane GQE demo takes over two hours for H2. Ours takes minutes, for
four reasons:

- **Small model.** 1.5M parameters, not 85M. The vocabulary is 12 tokens and
  sequences are 8 long; a large model is pure waste.
- **A purpose-built simulator.** Our circuits only ever use diagonal phases and
  a uniform X mixer. Both are one vectorised operation. A general SDK pays
  Python overhead per gate; we pay it once per batch. Notebook 03 checks our
  simulator against PennyLane and they agree.
- **A precomputed phase table.** Only a handful of `gamma` values exist, so
  `exp(-i*gamma*E)` is computed once and reused.
- **A cheap readout.** We only look at the K lowest-energy basis states, so
  reading the energy costs almost nothing.

### Size limits

Memory for one statevector is `2^n * 8` bytes, and we run a batch of 16 at once.

| qubits | one state | batch of 16 | verdict |
|--------|-----------|-------------|---------|
| 16     | 0.5 MB    | 8 MB        | instant |
| 20     | 8 MB      | 134 MB      | fine on CPU |
| 22     | 34 MB     | 537 MB      | GPU recommended |
| 24     | 134 MB    | 2.1 GB      | GPU only |
| 26+    | 537 MB    | 8.6 GB      | out of reach |

**Practical ceiling: about 20 qubits on CPU, 24 on GPU.** For the challenge's
44-base example that means minimum stem length 4 (10 qubits, instant) rather
than 3 (27 qubits, too big). Use `rna.limit_stems` to dial the count.

---

## Honest framing

Read this before writing any slide.

**Nested RNA folding is solved exactly in O(L^3) by classical dynamic
programming. There is no quantum speedup available for the problem as stated,
and we do not claim one.** The challenge brief says outperforming classical
methods is not the goal, and we take that at face value.

What this project is actually for:

1. **Validation.** We know the right answer, so we can check whether the
   quantum formulation is correct at all.
2. **Pseudoknots.** Crossing stems make the classical problem NP-hard and break
   dynamic programming. In our QUBO they are one boolean flag
   (`allow_pseudoknots=True`). Notebook 05.
3. **Multi-objective mRNA design.** Fold energy plus codon adaptation, GC
   content and uridine depletion. Extra objectives destroy the optimal
   substructure that dynamic programming needs, but are just more QUBO terms
   for us. This is the Moderna-relevant frontier.

Two more things we report rather than hide:

- **Encoding error.** The QUBO uses a two-body cluster expansion of the Turner
  model: `J[s,t] = E({s,t}) - h[s] - h[t]`. This is **exact for any pair** of
  stems, because stem energies are not additive (a lone stem pays a hairpin
  closure penalty that two nested stems pay only once). It stays approximate
  for three or more stems at once. Notebook 02 measures the residual gap and
  asserts pair-exactness. Whatever gap remains is a hard floor no optimiser can
  beat.
- **The classical ablation.** Notebook 04 runs the same GPT emitting bitstrings
  directly, with no quantum circuit. At 10 qubits the search space is 1024
  states and every method finds the optimum, so nothing is separated at this
  size. We say so.

---

## Reproducibility

- All random seeds are fixed and passed explicitly.
- `rna.set_vienna_defaults()` pins temperature, dangling-end model and lonely
  pair handling. Skip it and your energies will not match anyone else's.
- Only random and public sequences are used. No Moderna, patient, clinical or
  proprietary data, as required by the challenge brief.
- `requirements.txt` lists exact packages. ViennaRNA is the only non-obvious one.

---

## References

- Nakaji et al., *The generative quantum eigensolver (GQE) and its application
  for ground state search*, arXiv:2401.09253
- PennyLane demo, *Generative quantum eigensolver training using PennyLane data*
- Fox et al., *RNA folding using quantum computers*, PLOS Comput Biol 2022
  (the stem-per-qubit encoding we build on)
- Zaborniak et al., *A QUBO model of the RNA folding problem optimized by
  variational hybrid quantum annealing*, arXiv:2208.04367
- Lorenz et al., *ViennaRNA Package 2.0*
