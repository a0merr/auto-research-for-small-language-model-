# Autoresearch Program

## Goal
Minimize `val_bpb` (validation bits per byte) on TinyShakespeare within a 5-minute training budget.
Lower val_bpb = better language model.

## What You Can Modify
Only the HYPERPARAMETERS section in `train.py`. Specifically:
- `DEPTH` — number of transformer layers (2–8)
- `N_HEADS` — attention heads (must divide N_EMBD evenly)
- `N_EMBD` — embedding dimension (64, 128, 256, 512)
- `DROPOUT` — regularization (0.0–0.3)
- `SEQ_LEN` — context window (64–512)
- `BATCH_SIZE` — batch size (8–64, stay within 8GB VRAM)
- `LEARNING_RATE` — step size (1e-4 to 1e-3)
- `WEIGHT_DECAY` — L2 regularization (0.01–0.2)
- `WARMUP_STEPS` — LR warmup (50–200)

## Research Strategy
1. Start with baseline, understand the numbers
2. Try one change at a time — isolate what helps
3. If learning rate hasn't been explored, try that first (high impact)
4. If model is underfitting (loss still falling at end), increase capacity
5. If model is overfitting, increase dropout or decrease model size
6. Keep notes on what worked — build on successes

## Hardware
- GPU: NVIDIA RTX 5060 (~8GB VRAM)
- Keep total model parameters under 10M to stay safe on VRAM
- Larger batch sizes can help stability but use more memory

## Output Format
Always respond with:
1. REASONING: (why this change should help)
2. CHANGE: (one-line summary of the change)
3. CODE: (complete modified train.py)
