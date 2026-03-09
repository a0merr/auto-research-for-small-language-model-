"""
prepare.py -- One-time data preparation and shared utilities.
DO NOT MODIFY -- the agent never touches this file.

Downloads TinyShakespeare, trains a BPE tokenizer, saves train/val splits.
"""
import os
import sys
import json
import time
import math
import struct
import urllib.request
import numpy as np

# ── Fixed constants ──────────────────────────────────────────────────────────
DATA_URL      = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
RAW_FILE      = os.path.join(DATA_DIR, "input.txt")
TRAIN_FILE    = os.path.join(DATA_DIR, "train.bin")
VAL_FILE      = os.path.join(DATA_DIR, "val.bin")
TOKENIZER_FILE= os.path.join(DATA_DIR, "tokenizer.json")

VOCAB_SIZE    = 256          # byte-level tokenizer (simple & fast)
MAX_SEQ_LEN   = 256          # context window
EVAL_TOKENS   = 65536        # tokens used for validation loss
VAL_SPLIT     = 0.1          # 10% validation

# ── Data preparation ─────────────────────────────────────────────────────────

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        print(f"[prepare] Data already exists at {RAW_FILE}")
        return
    print(f"[prepare] Downloading TinyShakespeare...")
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    print(f"[prepare] Downloaded {os.path.getsize(RAW_FILE):,} bytes")

def encode(text):
    """Byte-level encoding."""
    return list(text.encode("utf-8"))

def decode(tokens):
    """Byte-level decoding."""
    return bytes(tokens).decode("utf-8", errors="replace")

def build_tokenizer():
    if os.path.exists(TOKENIZER_FILE):
        print("[prepare] Tokenizer already exists.")
        return
    print("[prepare] Building byte-level tokenizer...")
    tok = {"vocab_size": VOCAB_SIZE, "type": "byte_level"}
    with open(TOKENIZER_FILE, "w") as f:
        json.dump(tok, f)
    print(f"[prepare] Tokenizer saved (vocab_size={VOCAB_SIZE})")

def tokenize_data():
    if os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE):
        print("[prepare] Tokenized data already exists.")
        return
    print("[prepare] Tokenizing data...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = encode(text)
    n      = len(tokens)
    split  = int(n * (1 - VAL_SPLIT))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    def save_bin(path, toks):
        arr = np.array(toks, dtype=np.uint16)
        arr.tofile(path)

    save_bin(TRAIN_FILE, train_tokens)
    save_bin(VAL_FILE,   val_tokens)
    print(f"[prepare] Train tokens: {len(train_tokens):,}  Val tokens: {len(val_tokens):,}")

def load_tokenizer():
    with open(TOKENIZER_FILE) as f:
        return json.load(f)

# ── DataLoader ───────────────────────────────────────────────────────────────

class DataLoader:
    def __init__(self, path, seq_len, batch_size):
        self.data      = np.fromfile(path, dtype=np.uint16).astype(np.int64)
        self.seq_len   = seq_len
        self.batch_size= batch_size
        self.pos       = 0

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        buf  = self.data[self.pos: self.pos + B * T + 1]
        if len(buf) < B * T + 1:
            self.pos = 0
            buf = self.data[self.pos: self.pos + B * T + 1]
        x = buf[:-1].reshape(B, T)
        y = buf[1: ].reshape(B, T)
        self.pos += B * T
        return x, y

# ── Evaluation ───────────────────────────────────────────────────────────────

def estimate_loss(model, device, batch_size, seq_len, eval_tokens=EVAL_TOKENS):
    """Estimate validation loss. Returns bits-per-byte."""
    import torch
    model.eval()
    loader   = DataLoader(VAL_FILE, seq_len, batch_size)
    total_loss = 0.0
    n_batches  = max(1, eval_tokens // (batch_size * seq_len))
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = loader.next_batch()
            x    = torch.tensor(x, dtype=torch.long, device=device)
            y    = torch.tensor(y, dtype=torch.long, device=device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / n_batches
    bpb      = avg_loss / math.log(2)   # nats -> bits
    return bpb

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_data()
    build_tokenizer()
    tokenize_data()
    print("[prepare] All done! Run `python runner.py` to start autoresearch.")
