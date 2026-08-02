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
                         alpha=0.15, verbose=True):
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
                  lr=3e-4, device="cpu", log_every=50):
    """Plain supervised training on the fixed dataset."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tokens = torch.as_tensor(pools.to_gpt_tokens(tokens_np), device=device)
    energies = to_torch(energies_np, device)

    n = tokens.shape[0]
    history = []
    model.train()
    for ep in range(epochs):
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
        if log_every and (ep + 1) % log_every == 0:
            print(f"  offline epoch {ep + 1:4d}  loss {history[-1]:.4f}")
    return history


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
                 temp_min=0.05, temp_decay=0.985, buffer_size=600,
                 loss_kind="logit", alpha=0.15, log_every=25, time_budget=None):
    """The real GQE loop: sample -> score -> learn -> repeat.

    temperature controls exploration. It starts high (try lots of things) and
    decays (focus on what works). This plays the same role as the beta
    schedule in the paper.

    time_budget : optional seconds. Stops early so a Colab session cannot
                  time out on you halfway through.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    buf = ReplayBuffer(buffer_size)
    history = {"best": [], "mean_cvar": [], "temp": [], "loss": []}
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
            best_tokens = idx[int(np.argmin(out["best"][:, -1]))].copy()

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
        history["temp"].append(float(temperature))
        history["loss"].append(loss_val)

        if log_every and (ep + 1) % log_every == 0:
            print(f"  epoch {ep + 1:4d}  best {best_ever:8.3f}  "
                  f"mean CVaR {history['mean_cvar'][-1]:8.3f}  T {temperature:.3f}  "
                  f"({time.time() - t0:.0f}s)")

        if time_budget and (time.time() - t0) > time_budget:
            print(f"  stopped at epoch {ep + 1}: time budget reached")
            break

    return history, best_tokens, best_index
