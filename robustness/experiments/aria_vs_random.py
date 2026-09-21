"""
Milestone-3 exit condition (spec section 31): "ARIA obeys constraints and
optimized attack is stronger than the matched random radiometric control on
a validation subset."

Covers conditions 7-8 of the section-38 minimal experiment (ARIA at
weak/medium; random-radiometric-control comparison added at the same
severities as its mandatory matched baseline). Same protocol as
snua_vs_random.py: YOLO26n, NSLSR leakage-safe VAL split, best.pt checkpoint,
seeds 0/1/2, weak+medium severity.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import csv
import json
import statistics

from robustness.attacks.aria import ARIAAttack
from robustness.attacks.radiometric import RandomSmoothRadiometricAttack
from robustness.eval_runner import evaluate_attack

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"
SPLIT = "val"
SEEDS = [0, 1, 2]
SEVERITIES = ["weak", "medium"]


def _extract(stats):
    return {
        "precision": stats.get("metrics/precision(B)"),
        "recall": stats.get("metrics/recall(B)"),
        "mAP50": stats.get("metrics/mAP50(B)"),
        "mAP50-95": stats.get("metrics/mAP50-95(B)"),
    }


def main():
    rows = []

    print("=== clean (reference point, no seed dependence) ===")
    clean = _extract(evaluate_attack(MODEL_PATH, YAML_PATH, SPLIT, attack=None))
    print(clean)
    rows.append({"attack": "clean", "severity": None, "seed": None, **clean})
    clean_map50 = clean["mAP50"]

    for severity in SEVERITIES:
        for attack_name, attack_cls in [("random_radiometric", RandomSmoothRadiometricAttack), ("aria", ARIAAttack)]:
            per_seed_map50 = []
            for seed in SEEDS:
                attack = attack_cls.from_severity_name(severity)
                stats = _extract(evaluate_attack(MODEL_PATH, YAML_PATH, SPLIT, attack=attack, seed=seed))
                print(f"[{attack_name} | {severity} | seed={seed}] {stats}")
                rows.append({"attack": attack_name, "severity": severity, "seed": seed, **stats})
                per_seed_map50.append(stats["mAP50"])

            mean_map50 = statistics.mean(per_seed_map50)
            std_map50 = statistics.stdev(per_seed_map50) if len(per_seed_map50) > 1 else 0.0
            drop = clean_map50 - mean_map50
            print(f"  -> mean mAP50 = {mean_map50:.4f} (std {std_map50:.4f}), drop from clean = {drop:.4f}\n")

    csv_path = REPO_ROOT / "robustness_results_aria_vs_random_minimal.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "severity", "seed", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    print("\n=== Milestone-3 exit condition: ARIA stronger than matched random-radiometric control ===")
    exit_condition_met = {}
    for severity in SEVERITIES:
        rand_drops = [clean_map50 - r["mAP50"] for r in rows if r["attack"] == "random_radiometric" and r["severity"] == severity]
        aria_drops = [clean_map50 - r["mAP50"] for r in rows if r["attack"] == "aria" and r["severity"] == severity]
        mean_rand_drop = statistics.mean(rand_drops)
        mean_aria_drop = statistics.mean(aria_drops)
        met = mean_aria_drop > mean_rand_drop
        exit_condition_met[severity] = met
        print(f"  [{severity}] mean drop: random_radiometric={mean_rand_drop:.4f}, aria={mean_aria_drop:.4f} -> ARIA stronger: {met}")

    with open(REPO_ROOT / "robustness_results_aria_vs_random_summary.json", "w") as f:
        json.dump({"clean_mAP50": clean_map50, "exit_condition_met": exit_condition_met}, f, indent=2)

    return exit_condition_met


if __name__ == "__main__":
    main()
