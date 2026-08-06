"""A small GPT that writes quantum circuits.

This is a compact stand-in for nanoGPT. It is small on purpose: the demo in the
PennyLane tutorial uses 85 million parameters for a vocabulary of about 50
tokens, which is huge overkill and is why it took over two hours. Ours is about
1-2 million parameters and trains in minutes on a free Colab GPU.

Sign convention (easy to get backwards, so read this twice):
    the model outputs a LOGIT per token, and we train that logit to equal the
    ENERGY. So LOW logit = LOW energy = GOOD. When sampling we therefore use
    softmax(-logit / temperature).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        # Causal mask: a token may only look at earlier tokens.
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTQE(nn.Module):
    """Generates token sequences; each logit predicts an energy."""

    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4,
                 n_embd=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    # ---------------- losses ----------------

    def logit_matching_loss(self, tokens, energies):
        """Make the running sum of logits match the energy after each token.

        tokens   : (B, N+1) with the start token in front
        energies : (B, N) energy after each prefix
        """
        cur, nxt = tokens[:, :-1], tokens[:, 1:]
        logits = self(cur)
        picked = logits.gather(2, nxt.unsqueeze(-1)).squeeze(-1)   # (B, N)
        running = torch.cumsum(picked, dim=1)
        return F.mse_loss(running, energies)

    def grpo_loss(self, tokens, final_energy, eps=0.2):
        """GRPO: push up the probability of sequences that beat the batch average.

        The paper (v2) found this works better than logit matching, because it
        optimises the energy directly instead of just predicting it.
        """
        cur, nxt = tokens[:, :-1], tokens[:, 1:]
        logits = self(cur)
        logp = F.log_softmax(-logits, dim=-1)
        picked = logp.gather(2, nxt.unsqueeze(-1)).squeeze(-1)      # (B, N)

        reward = -final_energy
        adv = (reward - reward.mean()) / (reward.std() + 1e-8)
        ratio = torch.exp(picked - picked.detach())
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv.unsqueeze(1)
        return -clipped.mean()

    # ---------------- sampling ----------------

    @torch.no_grad()
    def generate(self, n_seq, seq_len, temperature=1.0, device="cpu"):
        """Sample circuits. Returns (tokens, predicted total energy)."""
        idx = torch.zeros((n_seq, 1), dtype=torch.long, device=device)
        total = torch.zeros((n_seq, 1), device=device)
        for _ in range(seq_len):
            ctx = idx[:, -self.block_size:]
            logits = self(ctx)[:, -1, :].clone()   # clone: do not touch the graph
            logits[:, 0] = float("inf")            # never re-emit the start token
            probs = F.softmax(-logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            total += logits.gather(1, nxt)
            idx = torch.cat([idx, nxt], dim=1)
        return idx, total


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_torch(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
