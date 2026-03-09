# 🔬 Autoresearch

An autonomous ML research system powered by Claude. It trains a small GPT language model, evaluates results, and uses Claude to decide what to try next — all night, while you sleep.

```
 Experiment 1  →  Train (5min)  →  Claude analyzes  →  Edits train.py
 Experiment 2  →  Train (5min)  →  Claude analyzes  →  Edits train.py
 ...repeat...
 Wake up to 20 experiments + a better model
```

---

## ✨ Features

- 🤖 **Claude as the agent** — explains its reasoning for every change
- 📊 **Live browser dashboard** — real-time loss curves, experiment history, Claude's reasoning
- 💻 **Windows native** — works out of the box in PowerShell
- 🎯 **One change at a time** — clean, reproducible experiments
- 📁 **Full history saved** — every experiment's config, results, and Claude's reasoning

---

## 🚀 Quick Start

### 1. Install dependencies
```powershell
pip install torch numpy anthropic
```

### 2. Set your Anthropic API key
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### 3. Run everything with one command
```powershell
.\start.ps1
```

This will:
- Download TinyShakespeare dataset
- Start the live dashboard at http://localhost:8080
- Run 20 experiments autonomously with Claude

---

## 🛠️ Manual Setup

If you prefer to run things separately:

```powershell
# 1. Prepare data (one-time)
python prepare.py

# 2. Start dashboard (in a separate terminal)
python dashboard/server.py

# 3. Run autoresearch
python runner.py --experiments 20
```

---

## 📁 Project Structure

```
autoresearch/
├── start.ps1            # One-click Windows launcher
├── runner.py            # Main orchestrator loop
├── agent.py             # Claude API integration
├── train.py             # GPT model (Claude modifies this)
├── prepare.py           # Data prep utilities (never modified)
├── program.md           # Your instructions to Claude
├── requirements.txt
├── dashboard/
│   ├── server.py        # Local web server
│   └── index.html       # Live dashboard UI
└── results/             # Per-experiment JSON results + train.py snapshots
```

---

## ⚙️ How It Works

1. **Runner** starts a training experiment (`train.py` runs for 5 minutes)
2. **Metrics** stream to the dashboard in real time (loss, val_bpb, progress)
3. **Claude** reads the full experiment history and `train.py`, reasons about what to change
4. **Agent** writes the new `train.py` (only the hyperparameters section)
5. Repeat — each experiment builds on what Claude learned

---

## 🎛️ Configuration

Edit `program.md` to change Claude's research strategy. Edit the top of `train.py` to set the baseline hyperparameters. Both files are designed to be human-readable and easy to tweak.

**Hyperparameters Claude can explore:**

| Parameter | Default | Range |
|-----------|---------|-------|
| DEPTH | 4 | 2–8 |
| N_HEADS | 4 | 2–8 |
| N_EMBD | 128 | 64–512 |
| SEQ_LEN | 128 | 64–512 |
| BATCH_SIZE | 16 | 8–64 |
| LEARNING_RATE | 3e-4 | 1e-4–1e-3 |
| DROPOUT | 0.1 | 0.0–0.3 |

---

## 💡 Tips

- **Let it run overnight** — 20 experiments × 5 minutes = ~2 hours
- **Check the dashboard** at http://localhost:8080 for live progress
- **Read Claude's reasoning** in the dashboard — it's genuinely interesting
- **Edit program.md** to steer the research in a direction you care about

---

## 📋 Requirements

- Python 3.9+
- NVIDIA GPU (RTX series recommended)
- PyTorch with CUDA
- Anthropic API key

---

## 📜 License

MIT
