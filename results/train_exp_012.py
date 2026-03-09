"""
train.py -- Small GPT model + training loop.
THIS FILE IS MODIFIED BY THE AGENT. All hyperparameters and architecture
choices are fair game. The agent reads results and edits this file.

Current config: Baseline small GPT
"""
import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from prepare import DataLoader, estimate_loss, TRAIN_FILE, VAL_FILE, VOCAB_SIZE, MAX_SEQ_LEN

# ── HYPERPARAMETERS (agent modifies these) ───────────────────────────────────

# Model architecture
DEPTH       = 4          # number of transformer layers
N_HEADS     = 4          # number of attention heads
N_EMBD      = 128        # embedding dimension
DROPOUT     = 0.1        # dropout rate
SEQ_LEN     = 128        # context window length

# Training
BATCH_SIZE      = 16     # batch size
LEARNING_RATE   = 3e-4   # learning rate
WEIGHT_DECAY    = 0.1    # weight decay
GRAD_CLIP       = 1.0    # gradient clipping
WARMUP_STEPS    = 100    # LR warmup steps

# Schedule
TRAIN_MINUTES   = 5      # total training time in minutes
EVAL_INTERVAL   = 50     # evaluate every N steps
LOG_INTERVAL    = 10     # log every N steps

# ── MODEL ────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, seq_len, dropout):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.n_embd  = n_embd
        self.head_dim= n_embd // n_heads
        self.qkv     = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj    = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                             .view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1  = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2  = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, seq_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn= CausalSelfAttention(n_embd, n_heads, seq_len, dropout)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, depth, seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            Block(n_embd, n_heads, seq_len, dropout) for _ in range(depth)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        pos    = torch.arange(T, device=idx.device)
        x      = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x  = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ── TRAINING LOOP ────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_lr, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

def train(results_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")

    model = GPT(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_heads=N_HEADS,
        depth=DEPTH,
        seq_len=SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    print(f"[train] Model params: {model.num_params():,}")
    print(f"[train] Config: depth={DEPTH} heads={N_HEADS} embd={N_EMBD} seq={SEQ_LEN} bs={BATCH_SIZE}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    loader     = DataLoader(TRAIN_FILE, SEQ_LEN, BATCH_SIZE)
    end_time   = time.time() + TRAIN_MINUTES * 60
    total_steps= int(TRAIN_MINUTES * 60 * 1000 / (SEQ_LEN * BATCH_SIZE / 1000))

    metrics    = []
    step       = 0
    best_bpb   = float('inf')

    while time.time() < end_time:
        lr = get_lr(step, WARMUP_STEPS, LEARNING_RATE, total_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr

        x, y = loader.next_batch()
        x    = torch.tensor(x, dtype=torch.long, device=device)
        y    = torch.tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            elapsed  = TRAIN_MINUTES * 60 - (end_time - time.time())
            progress = elapsed / (TRAIN_MINUTES * 60) * 100
            print(f"[train] step={step:5d} loss={loss.item():.4f} lr={lr:.2e} "
                  f"progress={progress:.1f}%")

        if step % EVAL_INTERVAL == 0:
            bpb = estimate_loss(model, device, BATCH_SIZE, SEQ_LEN)
            best_bpb = min(best_bpb, bpb)
            elapsed  = TRAIN_MINUTES * 60 - (end_time - time.time())
            print(f"[train] >>> EVAL step={step} val_bpb={bpb:.4f} best={best_bpb:.4f}")
            metrics.append({
                "step":    step,
                "val_bpb": round(bpb, 4),
                "loss":    round(loss.item(), 4),
                "elapsed": round(elapsed, 1),
            })
            # Write metrics for dashboard
            if results_file:
                with open(results_file, "w") as f:
                    json.dump({"metrics": metrics, "best_bpb": best_bpb,
                               "config": {
                                   "depth": DEPTH, "n_heads": N_HEADS,
                                   "n_embd": N_EMBD, "seq_len": SEQ_LEN,
                                   "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                                   "params": model.num_params(),
                               }}, f)
        step += 1

    print(f"[train] Done. Best val_bpb={best_bpb:.4f}")
    return best_bpb, metrics

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else None
    train(results_file)
