"""
Spec section 38: "Immediate minimal experiment" -- the exact 9-condition
table, run once as specified before scaling to YOLO26s/RASMD/training
defenses/transfer experiments.

Model: YOLO26n. Dataset: NSLSR leakage-safe. Checkpoint: fixed
validation-selected checkpoint (best.pt). Split: val. Seeds: 0,1,2.
Metrics: P, R, mAP50, mAP50-95.

Conditions 1,3,4,5,6 and 7,8 duplicate what snua_vs_random.py /
aria_vs_random.py already measured -- re-run here (cheap, ~1 min/condition)
so this script is the single self-contained source of the canonical table
required by section 38, rather than requiring the reader to stitch together
three separate result files.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import csv
import statistics

from robustness.attacks.aria import ARIAAttack
from robustness.attacks.combined import CombinedSNUAARIAAttack
from robustness.attacks.pgd import PGDAttack
from robustness.attacks.radiometric import RandomSmoothRadiometricAttack
from robustness.attacks.random_fpn import RandomFPNAttack
from robustness.attacks.snua import SNUAGoAttack
from robustness.eval_runner import evaluate_attack

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"
SPLIT = "val"
SEEDS = [0, 1, 2]

CONDITIONS = [
    ("1_clean", None, [None]),
    ("2_pgd20", lambda: PGDAttack(eps=8 / 255, alpha=2 / 255, steps=20, random_start=True), SEEDS),
    ("3_random_fpn_weak", lambda: RandomFPNAttack.from_severity_name("weak"), SEEDS),
    ("4_snua_go_weak", lambda: SNUAGoAttack.from_severity_name("weak"), SEEDS),
    ("5_random_fpn_medium", lambda: RandomFPNAttack.from_severity_name("medium"), SEEDS),
    ("6_snua_go_medium", lambda: SNUAGoAttack.from_severity_name("medium"), SEEDS),
    ("7_random_radiometric_medium", lambda: RandomSmoothRadiometricAttack.from_severity_name("medium"), SEEDS),
    ("8_aria_medium", lambda: ARIAAttack.from_severity_name("medium"), SEEDS),
    ("9_combined_snua_aria_medium", lambda: CombinedSNUAARIAAttack.from_severity_names("medium", "medium"), SEEDS),
]


def _extract(stats):
    return {
        "precision": stats.get("metrics/precision(B)"),
        "recall": stats.get("metrics/recall(B)"),
        "mAP50": stats.get("metrics/mAP50(B)"),
        "mAP50-95": stats.get("metrics/mAP50-95(B)"),
    }


def main():
    rows = []
    for cond_name, attack_factory, seeds in CONDITIONS:
        per_seed = []
        for seed in seeds:
            attack = attack_factory() if attack_factory is not None else None
            stats = _extract(evaluate_attack(MODEL_PATH, YAML_PATH, SPLIT, attack=attack, seed=seed))
            print(f"[{cond_name} | seed={seed}] {stats}")
            rows.append({"condition": cond_name, "seed": seed, **stats})
            per_seed.append(stats["mAP50"])
        mean_m = statistics.mean(per_seed)
        std_m = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        print(f"  -> {cond_name}: mean mAP50={mean_m:.4f} (std {std_m:.4f})\n")

    csv_path = REPO_ROOT / "minimal_experiment_section38_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
