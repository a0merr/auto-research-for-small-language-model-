"""
agent.py -- Claude-powered research agent.
Reads experiment results, reasons about what to try next,
and edits train.py with a new experiment.
"""
import os
import sys
import json
import re
import anthropic

TRAIN_FILE   = os.path.join(os.path.dirname(__file__), "train.py")
PROGRAM_FILE = os.path.join(os.path.dirname(__file__), "program.md")

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def build_prompt(experiment_history, current_train_py, program_md):
    history_str = ""
    for i, exp in enumerate(experiment_history):
        cfg = exp.get("config", {})
        history_str += f"""
Experiment {i+1}:
  Config: depth={cfg.get('depth')} heads={cfg.get('n_heads')} embd={cfg.get('n_embd')} seq={cfg.get('seq_len')} bs={cfg.get('batch_size')} lr={cfg.get('lr')} params={cfg.get('params'):,}
  Best val_bpb: {exp.get('best_bpb', 'N/A')}
  Reasoning: {exp.get('agent_reasoning', 'N/A')}
"""

    return f"""You are an autonomous ML research agent. Your goal is to improve a small GPT language model by modifying its training configuration.

## Your Instructions (program.md)
{program_md}

## Experiment History
{history_str if history_str else "No experiments yet - this is the first run."}

## Current train.py
```python
{current_train_py}
```

## Your Task
1. Analyze the experiment history and identify what's working and what isn't
2. Decide on ONE specific change to make (be focused, not multiple changes at once)
3. Explain your reasoning clearly
4. Output the modified train.py

## Rules
- Only modify the HYPERPARAMETERS section (between the dashed comment lines)
- Keep all other code exactly the same
- Make one meaningful change per experiment
- Aim to reduce val_bpb (lower is better)
- Consider: learning rate, depth, n_embd, n_heads, batch_size, dropout, seq_len
- Don't exceed GPU memory - RTX 5060 has ~8GB VRAM, keep model reasonable

## Output Format
Respond with:
1. REASONING: (2-3 sentences explaining your choice)
2. CHANGE: (one line describing exactly what you changed)
3. CODE: (the complete modified train.py between ```python and ```)
"""

def parse_response(response_text):
    reasoning = ""
    change    = ""
    new_code  = ""

    # Extract reasoning
    r_match = re.search(r"REASONING:\s*(.+?)(?=\n\d\.|\nCHANGE:|\nCODE:)", response_text, re.DOTALL)
    if r_match:
        reasoning = r_match.group(1).strip()

    # Extract change
    c_match = re.search(r"CHANGE:\s*(.+?)(?=\n\d\.|\nCODE:|\n```)", response_text, re.DOTALL)
    if c_match:
        change = c_match.group(1).strip()

    # Extract code block
    code_match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if code_match:
        new_code = code_match.group(1).strip()

    return reasoning, change, new_code

def run_agent(experiment_history, verbose=True):
    """
    Call Claude to analyze results and produce a new train.py.
    Returns (reasoning, change_description, success).
    """
    client = anthropic.Anthropic()

    current_train = load_file(TRAIN_FILE)
    program_md    = load_file(PROGRAM_FILE) if os.path.exists(PROGRAM_FILE) else "Minimize val_bpb. Try systematic hyperparameter exploration."

    prompt = build_prompt(experiment_history, current_train, program_md)

    if verbose:
        print("[agent] Asking Claude for next experiment...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text
    reasoning, change, new_code = parse_response(response_text)

    if verbose:
        print(f"[agent] Reasoning: {reasoning}")
        print(f"[agent] Change: {change}")

    if new_code:
        # Backup current train.py
        backup_path = TRAIN_FILE + ".backup"
        save_file(backup_path, current_train)
        # Write new train.py
        save_file(TRAIN_FILE, new_code)
        if verbose:
            print(f"[agent] Updated train.py (backup saved)")
        return reasoning, change, True
    else:
        if verbose:
            print("[agent] WARNING: Could not parse new code from Claude response")
        return reasoning, change, False

if __name__ == "__main__":
    # Test agent with empty history
    reasoning, change, success = run_agent([])
    print(f"Success: {success}")
    print(f"Reasoning: {reasoning}")
    print(f"Change: {change}")
