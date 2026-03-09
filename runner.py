"""
runner.py -- Autoresearch orchestrator.
Runs the train -> evaluate -> agent -> repeat loop.
Streams metrics to dashboard in real time.
"""
import os
import sys
import json
import time
import subprocess
import threading
import argparse
import shutil
from datetime import datetime

ROOT         = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(ROOT, "results")
DASHBOARD_DIR= os.path.join(ROOT, "dashboard")
STATE_FILE   = os.path.join(DASHBOARD_DIR, "state.json")
LOG_FILE     = os.path.join(ROOT, "autoresearch.log")

os.makedirs(RESULTS_DIR,   exist_ok=True)
os.makedirs(DASHBOARD_DIR, exist_ok=True)

def log(msg, also_print=True):
    ts  = datetime.now().strftime("%H:%M:%S")
    line= f"[{ts}] {msg}"
    if also_print:
        print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def write_state(state):
    """Write current state for dashboard."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def run_training(exp_id):
    """Run train.py as subprocess, stream output, return results."""
    result_file = os.path.join(RESULTS_DIR, f"exp_{exp_id:03d}.json")
    log(f"[runner] Starting experiment {exp_id}...")

    proc = subprocess.Popen(
    [sys.executable, os.path.join(ROOT, "train.py"), result_file],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace',
    cwd=ROOT,
)

    output_lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            line = line.rstrip()
            output_lines.append(line)
            log(f"  {line}", also_print=True)

            # Update dashboard with live progress
            if os.path.exists(result_file):
                try:
                    with open(result_file) as f:
                        partial = json.load(f)
                    partial["status"]  = "running"
                    partial["exp_id"]  = exp_id
                    partial["output"]  = output_lines[-20:]  # last 20 lines
                    write_state({"current": partial, "exp_id": exp_id})
                except Exception:
                    pass

    proc.wait()

    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)
        results["status"]   = "done"
        results["exp_id"]   = exp_id
        results["timestamp"]= datetime.now().isoformat()
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        return results
    else:
        log(f"[runner] WARNING: No results file found for exp {exp_id}")
        return None

def load_all_results():
    """Load all completed experiment results."""
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith(".json"):
            path = os.path.join(RESULTS_DIR, fname)
            try:
                with open(path) as f:
                    results.append(json.load(f))
            except Exception:
                pass
    return results

def save_train_snapshot(exp_id):
    """Save a snapshot of train.py for this experiment."""
    src = os.path.join(ROOT, "train.py")
    dst = os.path.join(RESULTS_DIR, f"train_exp_{exp_id:03d}.py")
    shutil.copy2(src, dst)

def build_history_for_agent(all_results):
    """Build experiment history summary for Claude."""
    history = []
    for r in all_results:
        history.append({
            "exp_id":         r.get("exp_id"),
            "best_bpb":       r.get("best_bpb"),
            "config":         r.get("config", {}),
            "agent_reasoning":r.get("agent_reasoning", ""),
            "agent_change":   r.get("agent_change", ""),
        })
    return history

def update_dashboard_state(all_results, current_exp=None, status="idle"):
    """Update the full dashboard state."""
    state = {
        "status":      status,
        "current_exp": current_exp,
        "total_exps":  len(all_results),
        "best_bpb":    min((r.get("best_bpb", 999) for r in all_results), default=None),
        "best_exp":    None,
        "history":     [],
        "updated_at":  datetime.now().isoformat(),
    }

    for r in all_results:
        state["history"].append({
            "exp_id":   r.get("exp_id"),
            "best_bpb": r.get("best_bpb"),
            "config":   r.get("config", {}),
            "reasoning":r.get("agent_reasoning", ""),
            "change":   r.get("agent_change", ""),
            "metrics":  r.get("metrics", []),
        })

    if all_results:
        best = min(all_results, key=lambda r: r.get("best_bpb", 999))
        state["best_exp"] = best.get("exp_id")

    write_state(state)

def main():
    parser = argparse.ArgumentParser(description="Autoresearch runner")
    parser.add_argument("--experiments", type=int, default=20,
                        help="Number of experiments to run (default: 20)")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip Claude agent (just run baseline repeatedly)")
    args = parser.parse_args()

    log("=" * 60)
    log("  AUTORESEARCH -- Claude-powered ML experimentation")
    log("=" * 60)
    log(f"  Experiments planned : {args.experiments}")
    log(f"  Results dir         : {RESULTS_DIR}")
    log(f"  Dashboard state     : {STATE_FILE}")
    log("=" * 60)

    # Check data exists
    data_dir = os.path.join(ROOT, "data")
    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        log("[runner] Data not found. Running prepare.py first...")
        subprocess.run([sys.executable, os.path.join(ROOT, "prepare.py")], cwd=ROOT)

    all_results = load_all_results()
    exp_id      = len(all_results) + 1

    update_dashboard_state(all_results, status="starting")

    for i in range(args.experiments):
        log(f"\n{'='*60}")
        log(f"  EXPERIMENT {exp_id} / {exp_id + args.experiments - i - 1} planned")
        log(f"{'='*60}")

        # Step 1: Ask Claude for next experiment (skip on first run)
        if not args.skip_agent and all_results:
            try:
                from agent import run_agent
                history  = build_history_for_agent(all_results)
                reasoning, change, success = run_agent(history)
                if not success:
                    log("[runner] Agent failed to produce new code, using current train.py")
                    reasoning, change = "Agent parse error - rerunning baseline", "none"
            except Exception as e:
                log(f"[runner] Agent error: {e} -- running with current train.py")
                reasoning, change = f"Error: {e}", "none"
        else:
            reasoning = "Baseline run" if not all_results else "Agent skipped"
            change    = "none"

        log(f"[runner] Agent reasoning: {reasoning}")
        log(f"[runner] Change made: {change}")

        # Step 2: Save train.py snapshot
        save_train_snapshot(exp_id)

        # Step 3: Run training
        update_dashboard_state(all_results, current_exp=exp_id, status="training")
        results = run_training(exp_id)

        if results:
            results["agent_reasoning"] = reasoning
            results["agent_change"]    = change
            # Save updated results with agent info
            result_file = os.path.join(RESULTS_DIR, f"exp_{exp_id:03d}.json")
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)

            all_results = load_all_results()
            best_so_far = min(r.get("best_bpb", 999) for r in all_results)

            log(f"[runner] Exp {exp_id} complete. val_bpb={results.get('best_bpb', 'N/A'):.4f}  "
                f"Best so far: {best_so_far:.4f}")
        else:
            log(f"[runner] Exp {exp_id} failed!")

        update_dashboard_state(all_results, status="between_experiments")
        exp_id += 1
        time.sleep(2)  # brief pause between experiments

    update_dashboard_state(all_results, status="done")
    log("\n[runner] All experiments complete!")
    log(f"[runner] Check dashboard at http://localhost:8080")
    log(f"[runner] Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
