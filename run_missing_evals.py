"""
Run every eval combination that is missing from the current results:
  (dataset x model x training x checkpoint) x attack x seed.

Streams one row per completed eval into `all_evals_results.csv` so a
kill / crash / power loss loses only the in-flight cell. On re-run,
already-present (dataset, model, training, checkpoint, eval_split, attack,
seed) tuples are skipped.

Attacks probed here are the FULL set (not just the D-matrix's medium
severities):
  clean, pgd20, snua_go_{weak,medium,strong}, aria_{weak,medium,strong},
  combined_{weak,medium,strong},
  random_fpn_{weak,medium}, random_radiometric_{weak,medium}.

Two orthogonal wins:
  * broader attack space  -> the D-matrix gets its "strong" column and a
    matched random control for every defense
  * new checkpoints as they land -> RASMD SWIR-attack matrix, small-model
    matrix, D6 combined-AT, D4 ARIA-AT
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import argparse
import csv
import time
import traceback

from robustness.attacks.aria import ARIAAttack
from robustness.attacks.combined import CombinedSNUAARIAAttack
from robustness.attacks.pgd import PGDAttack
from robustness.attacks.random_fpn import RandomFPNAttack
from robustness.attacks.radiometric import RandomSmoothRadiometricAttack
from robustness.attacks.snua import SNUAGoAttack
from robustness.eval_runner import evaluate_attack

OUT_CSV = REPO_ROOT / "all_evals_results.csv"
LOG     = REPO_ROOT / "all_evals_log.txt"

SEEDS_STOCH = [0, 1, 2]

# ---------- attack factory registry -------------------------------------
def _factory(name):
    if name == "clean":            return None
    if name == "pgd20":            return lambda: PGDAttack(eps=8/255, alpha=2/255, steps=20, random_start=True)
    if name.startswith("snua_go_"):    return lambda: SNUAGoAttack.from_severity_name(name.split("_")[-1])
    if name.startswith("aria_"):       return lambda: ARIAAttack.from_severity_name(name.split("_")[-1])
    if name.startswith("combined_"):
        sev = name.split("_")[-1]
        return lambda: CombinedSNUAARIAAttack.from_severity_names(sev, sev)
    if name.startswith("random_fpn_"):
        sev = name.split("_")[-1]
        return lambda: RandomFPNAttack.from_severity_name(sev)
    if name.startswith("random_radiometric_"):
        sev = name.split("_")[-1]
        return lambda: RandomSmoothRadiometricAttack.from_severity_name(sev)
    raise ValueError(f"unknown attack: {name}")


ATTACKS_STANDARD = [
    "clean", "pgd20",
    # SWIR attacks: keep weak+medium for SNUA (it's the primary attack;
    # weak is the training-time probe, medium the paper column), and add
    # strong for the failure-surface point. Drop weak for ARIA/combined --
    # ARIA-medium is already near-zero effect (0.4% drop on D0), so aria_weak
    # would just add cost without paper value.
    "snua_go_weak", "snua_go_medium", "snua_go_strong",
    "aria_medium", "aria_strong",
    "combined_medium", "combined_strong",
    # Matched random controls at MEDIUM only (the D-matrix's severity):
    # controls are for the "adversarial vs random-augmentation" argument
    # at the same severity the AT defenses were trained at.
    "random_fpn_medium",
    "random_radiometric_medium",
]


# ---------- checkpoint enumeration --------------------------------------
def enumerate_checkpoints():
    """Every trained checkpoint the repo has produced, tagged with the
    (dataset, split_variant, model, training, checkpoint) key so it lands
    in the same schema as results_summary_long.csv."""
    RT = REPO_ROOT / "runs" / "train"
    SAFE = REPO_ROOT / "dataset_safesplit.yaml"
    RAS  = REPO_ROOT / "dataset_rasmd.yaml"

    def _ck(run, ckpt):
        return run / "weights" / f"{ckpt}.pt"

    def _emit(dataset, split_variant, model, training, ckpts, run_dir, yaml_path):
        out = []
        if not run_dir.exists():
            return out
        for ckpt in ckpts:
            path = _ck(run_dir, ckpt)
            if path.exists():
                out.append({
                    "dataset": dataset, "split_variant": split_variant,
                    "model": model, "training": training, "checkpoint": ckpt,
                    "path": path, "yaml": yaml_path,
                })
        return out

    entries = []
    # NSLSR leakage-safe (segment)
    entries += _emit("NSLSR", "leakage-safe (segment)", "nano", "standard",   ["best", "last"], RT / "yolo26n_nslsr_safesplit",     SAFE)
    entries += _emit("NSLSR", "leakage-safe (segment)", "nano", "pgd_at",     ["best", "last"], RT / "yolo26n_nslsr_pgd_safesplit", SAFE)
    entries += _emit("NSLSR", "leakage-safe (segment)", "small","standard",   ["best", "last"], RT / "yolo26s_nslsr_safesplit",     SAFE)
    entries += _emit("NSLSR", "leakage-safe (segment)", "small","pgd_at",     ["best", "last"], RT / "yolo26s_nslsr_pgd_safesplit", SAFE)

    # NSLSR safesplit SWIR-domain defenses (D2/D3/D4/D5/D6, nano and small)
    for model, prefix in [("nano", "yolo26n"), ("small", "yolo26s")]:
        for training, suffix in [
            ("random_swir_aug", "d2_randaug"), ("snua_at", "d3_snua_at"),
            ("aria_at", "d4_aria"),           ("mixed_swir_at", "d5_mixed"),
            ("combined_at", "d6_combined_at"),
        ]:
            run = RT / f"{prefix}_nslsr_{suffix}"
            entries += _emit("NSLSR", "leakage-safe (segment)", model, training,
                             ["robust_best"], run, SAFE)

    # SWIR-defense checkpoints -- ONLY report the checkpoint predeclared by
    # the §13.3 selection rule (robust_best.pt). Ultralytics also writes
    # best.pt/last.pt, but evaluating them here would (a) triple the eval
    # cost with no reporting purpose, and (b) blur the selection story --
    # so they are intentionally skipped.

    # RASMD stratified (own)
    entries += _emit("RASMD", "stratified (own)", "nano", "standard", ["best", "last"], RT / "yolo26n_rasmd",     RAS)
    entries += _emit("RASMD", "stratified (own)", "nano", "pgd_at",   ["best", "last"], RT / "yolo26n_rasmd_pgd", RAS)
    entries += _emit("RASMD", "stratified (own)", "small","standard", ["best", "last"], RT / "yolo26s_rasmd",     RAS)
    entries += _emit("RASMD", "stratified (own)", "small","pgd_at",   ["best", "last"], RT / "yolo26s_rasmd_pgd", RAS)
    for model, prefix in [("nano", "yolo26n"), ("small", "yolo26s")]:
        for training, suffix in [
            ("random_swir_aug", "d2_randaug"), ("snua_at", "d3_snua_at"),
            ("mixed_swir_at", "d5_mixed"),
        ]:
            run = RT / f"{prefix}_rasmd_{suffix}"
            entries += _emit("RASMD", "stratified (own)", model, training,
                             ["robust_best"], run, RAS)

    return entries


# ---------- already-done bookkeeping ------------------------------------
FIELDS = [
    "dataset", "split_variant", "model", "training", "checkpoint",
    "eval_split", "attack", "seed",
    "precision", "recall", "mAP50", "mAP50-95",
    "elapsed_s", "timestamp",
]

def load_done():
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["dataset"], r["split_variant"], r["model"], r["training"],
                       r["checkpoint"], r["eval_split"], r["attack"], r["seed"])
                done.add(key)
    return done


def append_row(row):
    exists = OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ---------- main loop ---------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--attacks", nargs="+", default=ATTACKS_STANDARD)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="filter by dataset (NSLSR, RASMD); default all")
    p.add_argument("--models", nargs="+", default=None,
                   help="filter by model (nano, small); default all")
    args = p.parse_args()

    done = load_done()
    entries = enumerate_checkpoints()
    if args.datasets:
        entries = [e for e in entries if e["dataset"] in args.datasets]
    if args.models:
        entries = [e for e in entries if e["model"] in args.models]

    # count work
    todo = []
    for e in entries:
        for atk in args.attacks:
            seeds = [None] if atk == "clean" else SEEDS_STOCH
            for seed in seeds:
                key = (e["dataset"], e["split_variant"], e["model"], e["training"],
                       e["checkpoint"], args.split, atk, "" if seed is None else str(seed))
                if key in done:
                    continue
                todo.append((e, atk, seed))

    log(f"\n===== run_missing_evals starting =====")
    log(f"  checkpoints discovered: {len(entries)}")
    log(f"  eval_split: {args.split}   attacks: {len(args.attacks)}")
    log(f"  cells done already:    {len(done)}")
    log(f"  cells to run this run: {len(todo)}\n")

    for i, (e, atk, seed) in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {e['dataset']}|{e['model']}|{e['training']}|{e['checkpoint']}|{atk}|seed={seed}"
        t0 = time.time()
        try:
            attack = _factory(atk)
            attack = attack() if attack is not None else None
            stats = evaluate_attack(e["path"], e["yaml"], args.split,
                                    attack=attack, seed=seed)
            row = {
                "dataset": e["dataset"], "split_variant": e["split_variant"],
                "model": e["model"], "training": e["training"], "checkpoint": e["checkpoint"],
                "eval_split": args.split, "attack": atk,
                "seed": "" if seed is None else seed,
                "precision": stats.get("metrics/precision(B)"),
                "recall":    stats.get("metrics/recall(B)"),
                "mAP50":     stats.get("metrics/mAP50(B)"),
                "mAP50-95":  stats.get("metrics/mAP50-95(B)"),
                "elapsed_s": f"{time.time()-t0:.1f}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            append_row(row)
            log(f"{tag} -> mAP50={row['mAP50']:.4f} ({row['elapsed_s']}s)")
        except Exception as ex:
            log(f"{tag} -> ERROR: {ex}")
            traceback.print_exc()

    log(f"\n===== run_missing_evals done ({len(todo)} cells) =====\n")


if __name__ == "__main__":
    main()
