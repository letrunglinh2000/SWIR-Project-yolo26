"""
End-to-end driver. Four phases, resumable, streams progress to
`run_all_missing_master.log`.

  1) evals on existing checkpoints, TEST split (fast, high value)
  2) trainings for every missing SWIR defense (slow, ~26 GPU-h)
  3) evals on the new checkpoints just trained (fast)
  4) diagnostics + aggregators

Between (2) and (3), the new checkpoint dirs land under runs/train/;
run_missing_evals.py picks them up automatically because its
enumerate_checkpoints() glob-checks the disk each launch.
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
MASTER_LOG = REPO / "run_all_missing_master.log"


def _log(msg):
    with open(MASTER_LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _phase(name, argv):
    _log(f"\n\n########## {time.strftime('%Y-%m-%d %H:%M:%S')}  {name} ##########\n")
    t0 = time.time()
    proc = subprocess.run(argv, cwd=str(REPO))
    dt = time.time() - t0
    _log(f"\n{name} exit={proc.returncode}  elapsed={dt/3600:.2f}h")
    return proc.returncode


def main():
    py = sys.executable

    # 1) evals on everything that exists NOW
    _phase("PHASE 1: evals (existing checkpoints, TEST split)",
           [py, "run_missing_evals.py", "--split", "test"])

    # 2) trainings
    _phase("PHASE 2: run every missing training",
           [py, "run_missing_trainings.py"])

    # 3) evals again -- new checkpoints get picked up automatically because
    # enumerate_checkpoints() re-scans the disk each launch; the dedup on
    # (dataset,model,training,checkpoint,eval_split,attack,seed) skips every
    # cell phase 1 already wrote, so this run only pays for the new rows.
    # val is intentionally skipped: test is the reporting split, and the
    # D-matrix already has val rows for the existing checkpoints. Reader can
    # add --split val later for the new models if it becomes paper-relevant.
    _phase("PHASE 3: evals on the new checkpoints, TEST split",
           [py, "run_missing_evals.py", "--split", "test"])

    # 4) diagnostic scripts
    _phase("PHASE 4a: ARIA strong-severity check",
           [py, "robustness/experiments/aria_strong_check.py"])

    # 4b) rebuild the aggregate CSVs
    _phase("PHASE 4b: rebuild aggregate + summary CSVs",
           [py, "aggregate_results_table3.py"])
    _phase("PHASE 4c: rebuild results_summary CSVs",
           [py, "summarize_all_results.py"])

    _log(f"\n\n===== ALL PHASES COMPLETE ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====")


if __name__ == "__main__":
    main()
