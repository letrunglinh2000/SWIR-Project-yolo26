"""
Transfer-robustness check: does the ALREADY-TRAINED PGD-adversarial
checkpoint (yolo26n_nslsr_pgd_safesplit -- L-infinity PGD-AT, trained
against pixel-ball perturbations) provide any protection against the
SWIR-specific attacks (SNUA-GO, ARIA, Combined) it was never trained
against?

This is pure INFERENCE against an existing checkpoint (no new training) --
run before deciding whether Milestone 5 (SNUA-AT / ARIA-AT / mixed SWIR-AT)
is necessary, and to see which attack family a from-scratch SWIR-aware
defense should prioritize.

Compares two checkpoints side by side, same attacks/severities/seeds:
  - yolo26n_nslsr_safesplit       (standard training)
  - yolo26n_nslsr_pgd_safesplit   (existing PGD-adversarial training)
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
from robustness.attacks.snua import SNUAGoAttack
from robustness.eval_runner import evaluate_attack

YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"
SPLIT = "val"
SEEDS = [0, 1, 2]

CHECKPOINTS = {
    "standard": REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt",
    "pgd_adversarial": REPO_ROOT / "runs/train/yolo26n_nslsr_pgd_safesplit/weights/best.pt",
}

ATTACKS = [
    ("clean", None, [None]),
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
    rows = []
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        for attack_name, attack_factory, seeds in ATTACKS:
            per_seed_map50 = []
            for seed in seeds:
                attack = attack_factory() if attack_factory is not None else None
                stats = _extract(evaluate_attack(ckpt_path, YAML_PATH, SPLIT, attack=attack, seed=seed))
                print(f"[{ckpt_name} | {attack_name} | seed={seed}] {stats}")
                rows.append({"checkpoint": ckpt_name, "attack": attack_name, "seed": seed, **stats})
                per_seed_map50.append(stats["mAP50"])
            mean_m = statistics.mean(per_seed_map50)
            print(f"  -> {ckpt_name}/{attack_name}: mean mAP50={mean_m:.4f}\n")

    csv_path = REPO_ROOT / "pgd_at_transfer_check_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "attack", "seed", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
