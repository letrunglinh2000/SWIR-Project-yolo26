"""
Milestone-5 defense evaluation matrix (spec section 24 + 13.3 checkpoint policy).

Checkpoint selection per defense (spec 13.3: "Do not choose best.pt vs
last.pt based on test robustness. Implement validation-time checkpoint
selection using a predeclared criterion."):

  D0 (standard, yolo26n_nslsr_safesplit)      -> best.pt
      No robustness objective during training -- clean-fitness selection
      (Ultralytics' own default) is the correct, and only sensible, rule.
  D1 (existing PGD-AT, yolo26n_nslsr_pgd_safesplit) -> last.pt
      Per this project's own prior finding (memory: best.pt vs last.pt
      comparison on the leakage-safe split): last.pt is substantially more
      PGD-20-robust than best.pt for this adversarially-trained checkpoint
      (best.pt is selected by Ultralytics' clean-fitness criterion, which
      is exactly the anti-pattern section 13.3 warns against). Predeclared
      BEFORE this evaluation, from a result already on record, not fit to
      the outcome here.
  D2/D3/D5 (this session's new defenses) -> robust_best.pt
      Selected DURING training by RobustCheckpointSelector's own rule
      (S = 0.5*mAP_clean + 0.5*mAP_robust, tracked every epoch) --
      the section-13.3-compliant selection this session built specifically
      to avoid the D1 anti-pattern going forward.

Evaluated against: clean, PGD-20 (existing generic baseline), SNUA-GO
(weak+medium), ARIA (medium), Combined (medium) -- on NSLSR leakage-safe
VAL split (never test), >=3 seeds for every stochastic attack.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import argparse
import csv

from robustness.attacks.aria import ARIAAttack
from robustness.attacks.combined import CombinedSNUAARIAAttack
from robustness.attacks.pgd import PGDAttack
from robustness.attacks.snua import SNUAGoAttack
from robustness.eval_runner import evaluate_attack

YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"
SPLIT = "val"          # default: the selection/dev split (never test) -- see --split
SEEDS = [0, 1, 2]

CHECKPOINTS = {
    "D0_standard":       REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt",
    "D1_pgd_at":         REPO_ROOT / "runs/train/yolo26n_nslsr_pgd_safesplit/weights/last.pt",
    "D2_random_swir_aug": REPO_ROOT / "runs/train/yolo26n_nslsr_d2_randaug/weights/robust_best.pt",
    "D3_snua_at":        REPO_ROOT / "runs/train/yolo26n_nslsr_d3_snua_at/weights/robust_best.pt",
    "D5_mixed_swir_at":  REPO_ROOT / "runs/train/yolo26n_nslsr_d5_mixed/weights/robust_best.pt",
}

CONDITIONS = [
    ("clean", None, [None]),
    ("pgd20", lambda: PGDAttack(eps=8/255, alpha=2/255, steps=20, random_start=True), SEEDS),
    ("snua_go_weak", lambda: SNUAGoAttack.from_severity_name("weak"), SEEDS),
    ("snua_go_medium", lambda: SNUAGoAttack.from_severity_name("medium"), SEEDS),
    ("aria_medium", lambda: ARIAAttack.from_severity_name("medium"), SEEDS),
    ("combined_medium", lambda: CombinedSNUAARIAAttack.from_severity_names("medium", "medium"), SEEDS),
]


def _extract(stats):
    return {
        "precision": stats.get("metrics/precision(B)"),
        "recall": stats.get("metrics/recall(B)"),
        "mAP50": stats.get("metrics/mAP50(B)"),
        "mAP50-95": stats.get("metrics/mAP50-95(B)"),
    }


def main():
    p = argparse.ArgumentParser()
    # `val` is the development default. `test` is legitimate ONLY as a
    # one-time final-report pass: every checkpoint rule is already frozen in
    # defense_matrix_checkpoint_rule.json, and D2/D3/D5's robust_best.pt was
    # SELECTED on val -- so val numbers for those three are selection-biased
    # and the held-out test pass is what is comparable to the PGD-era
    # test-split table (aggregate_results_table2.csv).
    p.add_argument("--split", default=SPLIT, choices=["val", "test"])
    p.add_argument("--out", default=None, help="output CSV (default: defense_matrix_results[_<split>].csv)")
    args = p.parse_args()
    split = args.split

    rows = []
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_name}: {ckpt_path} not found")
            continue
        for cond_name, attack_factory, seeds in CONDITIONS:
            for seed in seeds:
                attack = attack_factory() if attack_factory is not None else None
                stats = _extract(evaluate_attack(ckpt_path, YAML_PATH, split, attack=attack, seed=seed))
                print(f"[{ckpt_name} | {cond_name} | seed={seed}] {stats}")
                rows.append({"checkpoint": ckpt_name, "condition": cond_name, "seed": seed, "split": split, **stats})

    default_name = "defense_matrix_results.csv" if split == "val" else f"defense_matrix_results_{split}.csv"
    csv_path = Path(args.out) if args.out else REPO_ROOT / default_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "condition", "seed", "split", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
