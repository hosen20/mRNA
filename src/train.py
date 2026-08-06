"""Training loops for GPT-QE.

Two phases, same as the paper:

  offline : learn from a fixed pile of random circuits. Cheap, no feedback.
  online  : the model proposes circuits, we score them, it learns from its own
            output. This is where the real improvement happens.
"""

import time
import numpy as np
import torch

from . import pools
from .gpt import to_torch


def make_offline_dataset(sim, n_seq, seq_len, batch=16, seed=0,
                         alpha=None, verbose=True):
    """Score a pile of random circuits. This is the offline training set."""
    rng = np.random.default_rng(seed)
    idx = pools.random_token_sequences(sim.pool["size"], n_seq, seq_len, rng)
    cvars = []
    t0 = time.time()
    for start in range(0, n_seq, batch):
        chunk = idx[start:start + batch]
        cvars.append(sim.run(chunk, alpha=alpha)["cvar"])
        if verbose and (start // batch) % 10 == 0:
            print(f"  scored {start + len(chunk)}/{n_seq}  ({time.time() - t0:.0f}s)")
    return idx, np.concatenate(cvars, axis=0)


def train_offline(model, tokens_np, energies_np, epochs=300, batch=64,
                  lr=3e-4, device="cpu", log_every=50,
                  sim=None, seq_len=None, eval_every=0, n_eval=8):
    """Plain supervised training on the fixed dataset.

    The loss alone does not tell you much. It says the model can predict the
    energies of RANDOM circuits, not that it can produce good ones. Pass sim
    and seq_len with eval_every > 0 and we also sample circuits from the model
    every so often and score them for real. That is the number you care about.

    Returns (loss_history, eval_history) where eval_history is a list of
    (epoch, mean CVaR, best) for the generated circuits.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tokens = torch.as_tensor(pools.to_gpt_tokens(tokens_np), device=device)
    energies = to_torch(energies_np, device)

    n = tokens.shape[0]
    history, evals = [], []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            opt.zero_grad()
            loss = model.logit_matching_loss(tokens[sel], energies[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(sel)
        history.append(total / n)

        do_eval = (sim is not None and seq_len and eval_every
                   and ((ep + 1) % eval_every == 0 or ep == 0))
        if do_eval:
            model.eval()
            with torch.no_grad():
                tok, _ = model.generate(n_eval, seq_len, temperature=1.0,
                                        device=device)
            idx = pools.from_gpt_tokens(tok.cpu().numpy())
            out = sim.run(idx, record_prefix=False)
            per_circuit_best = out["best"].min(axis=1)
            tol = 1e-6 + 1e-3 * abs(sim.E_opt)
            evals.append((ep + 1, float(out["cvar"].mean()),
                          float(per_circuit_best.min()),
                          float((per_circuit_best < 0).mean()),
                          float((per_circuit_best <= sim.E_opt + tol).mean())))

        if log_every and (ep + 1) % log_every == 0:
            msg = f"  offline epoch {ep + 1:4d}  loss {history[-1]:10.4f}"
            if evals and evals[-1][0] == ep + 1:
                msg += (f"   CVaR {evals[-1][1]:7.2f}"
                        f"  opt_rate {evals[-1][4]:6.1%}"
                        f"  valid {evals[-1][3]:5.1%}")   # best reported at the end
            print(msg)

    if evals:
        best_seen = min(e[2] for e in evals)
        print()
        print("  " + "-" * 52)
        print(f"  offline training finished: {epochs} epochs")
        print(f"  loss           : {history[0]:.1f} -> {history[-1]:.1f}")
        opt_txt = f"   (optimum {sim.E_opt:.3f})" if sim is not None else ""
        print(f"  best energy    : {best_seen:.3f}{opt_txt}")
        print(f"  CVaR           : {evals[0][1]:.2f} -> {evals[-1][1]:.2f}")
        print(f"  opt_rate       : {evals[0][4]:.1%} -> {evals[-1][4]:.1%}")
        print("  " + "-" * 52)

    model.was_trained = True
    return history, evals


class ReplayBuffer:
    """Keeps the most recent circuits. First in, first out."""

    def __init__(self, max_size=600):
        self.max_size = max_size
        self.tokens = None
        self.energies = None

    def add(self, tokens, energies):
        if self.tokens is None:
            self.tokens, self.energies = tokens, energies
        else:
            self.tokens = np.concatenate([self.tokens, tokens], axis=0)
            self.energies = np.concatenate([self.energies, energies], axis=0)
        if len(self.tokens) > self.max_size:
            self.tokens = self.tokens[-self.max_size:]
            self.energies = self.energies[-self.max_size:]

    def __len__(self):
        return 0 if self.tokens is None else len(self.tokens)


def train_online(model, sim, seq_len, epochs=200, n_sample=16, batch=48,
                 n_iter=3, lr=3e-4, device="cpu", temperature=1.0,
                 temp_min=0.05, temp_decay=None, buffer_size=600,
                 loss_kind="logit", alpha=None, log_every=25, time_budget=None):
    """The real GQE loop: sample -> score -> learn -> repeat.

    temperature controls exploration. It starts high (try lots of things) and
    decays (focus on what works). This plays the same role as the beta
    schedule in the paper.

    time_budget : optional seconds. Stops early so a Colab session cannot
                  time out on you halfway through.
    """
    # Set the decay so exploration lasts most of the run. A fixed 0.985 hits
    # the floor after ~200 epochs, so a 2000 epoch run would spend 90% of its
    # time with no exploration at all.
    if temp_decay is None:
        temp_decay = (temp_min / temperature) ** (1.0 / max(1, int(0.8 * epochs)))
        print(f"  temp_decay set to {temp_decay:.4f} "
              f"(reaches {temp_min} at epoch {int(0.8*epochs)})")

    if not getattr(model, "was_trained", False):
        print("  !! This model has not been through offline training.")
        print("     If you restarted the session, re-run the offline cell first,")
        print("     or load results/gqe_model.pt. Online from scratch is far slower.")
    if epochs < 200:
        print(f"  !! epochs={epochs} is too few to see a trend through CVaR noise."
              f" Use at least 400.")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    buf = ReplayBuffer(buffer_size)
    # Why several metrics:
    #   best        running minimum. An extreme value. Never rises, so it
    #               saturates and stops being informative.
    #   mean_cvar   average over the batch. Noisy with a small batch.
    #   valid_rate  fraction of circuits finding ANY valid structure (E < 0).
    #               Saturates at 100% quickly, so it is only useful early.
    #   opt_rate    fraction of circuits that find the OPTIMUM. Cannot
    #               saturate until the model is perfect. THIS is the one to
    #               watch: it goes from a few percent toward 100%.
    history = {"best": [], "best_epoch": [], "mean_cvar": [], "temp": [],
               "loss": [], "cvar_smooth": [], "hit_rate": [], "opt_rate": []}
    tol = 1e-6 + 1e-3 * abs(sim.E_opt)
    best_ever, best_tokens, best_index = np.inf, None, None
    t0 = time.time()

    for ep in range(epochs):
        # 1. Model proposes circuits.
        model.eval()
        tok, _ = model.generate(n_sample, seq_len, temperature, device)
        idx = pools.from_gpt_tokens(tok.cpu().numpy())

        # 2. Score them on the simulator.
        out = sim.run(idx, alpha=alpha)
        buf.add(idx, out["cvar"])

        if out["best_energy"] < best_ever:
            best_ever = out["best_energy"]
            best_index = out["best_index"]
            # use the row that produced best_index, so the reported circuit
            # and the reported state actually match
            best_tokens = idx[out.get("best_row", 0)].copy()

        # 3. Learn from the buffer.
        model.train()
        tokens_t = torch.as_tensor(pools.to_gpt_tokens(buf.tokens), device=device)
        energies_t = to_torch(buf.energies, device)
        loss_val = 0.0
        for _ in range(n_iter):
            sel = torch.randint(0, len(buf), (min(batch, len(buf)),), device=device)
            opt.zero_grad()
            if loss_kind == "grpo":
                loss = model.grpo_loss(tokens_t[sel], energies_t[sel][:, -1])
            else:
                loss = model.logit_matching_loss(tokens_t[sel], energies_t[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_val += loss.item() / n_iter

        temperature = max(temp_min, temperature * temp_decay)
        history["best"].append(float(best_ever))
        history["mean_cvar"].append(float(out["cvar"][:, -1].mean()))
        per_circuit_best = out["best"].min(axis=1)      # best per circuit
        history["best_epoch"].append(float(per_circuit_best.min()))
        history["hit_rate"].append(float((per_circuit_best < 0).mean()))
        history["opt_rate"].append(
            float((per_circuit_best <= sim.E_opt + tol).mean()))
        history["temp"].append(float(temperature))
        history["loss"].append(loss_val)
        # CVaR bounces a lot with a small batch, so also keep a running mean.
        # Judge progress by THIS column, not by single epochs.
        w = history["mean_cvar"][-25:]
        history["cvar_smooth"].append(float(np.mean(w)))

        if log_every and (ep + 1) % log_every == 0:
            hr25 = float(np.mean(history["hit_rate"][-25:]))
            or25 = float(np.mean(history["opt_rate"][-25:]))
            # 'best' is deliberately NOT printed here. It is a running minimum,
            # so it freezes the moment any circuit touches the optimum and then
            # looks like a flat line pretending to be a training curve. It is
            # the final answer, not a progress metric, so it is reported once
            # at the end.
            print(f"  epoch {ep + 1:4d}  CVaR25 {history['cvar_smooth'][-1]:7.2f}"
                  f"  opt_rate25 {or25:6.1%}  valid {hr25:5.1%}"
                  f"  T {temperature:.3f}  ({time.time() - t0:.0f}s)")

        if time_budget and (time.time() - t0) > time_budget:
            print(f"  stopped at epoch {ep + 1}: time budget reached")
            break

    ran = len(history["best"])
    print()
    print("  " + "-" * 52)
    print(f"  online training finished: {ran} epochs, {time.time() - t0:.0f}s")
    print(f"  best energy found      : {best_ever:.3f}"
          f"   (optimum {sim.E_opt:.3f})")
    if best_ever <= sim.E_opt + 1e-6 + 1e-3 * abs(sim.E_opt):
        print("  -> reached the optimum")
    else:
        print(f"  -> short of the optimum by {best_ever - sim.E_opt:.3f}")
    print(f"  CVaR, first 25 epochs  : {np.mean(history['mean_cvar'][:25]):.3f}")
    print(f"  CVaR, last 25 epochs   : {np.mean(history['mean_cvar'][-25:]):.3f}")
    print(f"  opt_rate, first 25     : {np.mean(history['opt_rate'][:25]):.1%}")
    print(f"  opt_rate, last 25      : {np.mean(history['opt_rate'][-25:]):.1%}")
    print("  " + "-" * 52)

    return history, best_tokens, best_index
