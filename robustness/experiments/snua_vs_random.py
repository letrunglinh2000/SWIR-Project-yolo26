"""
Milestone-2 exit condition (spec section 31): "SNUA obeys constraints and
optimized attack is stronger than the matched random baseline on a
validation subset."

Also covers conditions 3-6 of the section-38 minimal experiment:
    3. Random FPN, weak     4. SNUA-GO, weak
    5. Random FPN, medium   6. SNUA-GO, medium

Model: YOLO26n. Dataset: NSLSR leakage-safe, VAL split (never test -- spec
section 1.4: no test-set attack/model selection). Checkpoint: best.pt
(validation-mAP-selected, the only frozen selection rule available before
Milestone 5's robust-training checkpoint policy exists).
Seeds: 0, 1, 2 (>=3 per spec section 38).

Per-seed, per-severity evaluation runs the FULL val split once per
(attack, seed) combination -- not a single batch -- so the mAP drop is a
real aggregate over the validation set, not a per-batch loss snapshot.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import json
import statistics

from robustness.attacks.random_fpn import RandomFPNAttack
from robustness.attacks.snua import SNUAGoAttack
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
        for attack_name, attack_cls in [("random_fpn", RandomFPNAttack), ("snua_go", SNUAGoAttack)]:
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
            print(f"  -> mean mAP50 = {mean_map50:.4f} (std {std_map50:.4f}), "
                  f"drop from clean = {drop:.4f}\n")

    # Write canonical long-format CSV
    import csv
    csv_path = REPO_ROOT / "robustness_results_snua_vs_random_minimal.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "severity", "seed", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    # Milestone-2 exit-condition check: SNUA-GO drop > Random FPN drop, per severity
    print("\n=== Milestone-2 exit condition: SNUA-GO stronger than matched Random FPN control ===")
    exit_condition_met = {}
    for severity in SEVERITIES:
        rand_drops = [clean_map50 - r["mAP50"] for r in rows if r["attack"] == "random_fpn" and r["severity"] == severity]
        snua_drops = [clean_map50 - r["mAP50"] for r in rows if r["attack"] == "snua_go" and r["severity"] == severity]
        mean_rand_drop = statistics.mean(rand_drops)
        mean_snua_drop = statistics.mean(snua_drops)
        met = mean_snua_drop > mean_rand_drop
        exit_condition_met[severity] = met
        print(f"  [{severity}] mean drop: random_fpn={mean_rand_drop:.4f}, snua_go={mean_snua_drop:.4f} "
              f"-> SNUA-GO stronger: {met}")

    with open(REPO_ROOT / "robustness_results_snua_vs_random_summary.json", "w") as f:
        json.dump({"clean_mAP50": clean_map50, "exit_condition_met": exit_condition_met}, f, indent=2)

    return exit_condition_met


if __name__ == "__main__":
    main()
