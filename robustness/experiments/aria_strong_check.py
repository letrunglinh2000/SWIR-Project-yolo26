"""
Follow-up to aria_vs_random.py: the medium-severity ARIA effect (~0.6% mAP50
drop) is far weaker than SNUA-GO's at matched severity (~45% drop). Per spec
section 37's own caution ("If SNUA or ARIA cannot satisfy condition 3 under
reasonable budgets... investigate another physically motivated
parameterization"), check the "strong" severity before concluding ARIA is a
weak adversarial direction here -- rules out "budget too small" as the
explanation before flagging ARIA as a limitation.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import statistics

from robustness.attacks.aria import ARIAAttack
from robustness.attacks.radiometric import RandomSmoothRadiometricAttack
from robustness.eval_runner import evaluate_attack

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"
SPLIT = "val"
SEEDS = [0, 1, 2]


def _extract(stats):
    return {k: stats.get(f"metrics/{k}(B)") if k != "mAP50-95" else stats.get("metrics/mAP50-95(B)")
            for k in ["precision", "recall", "mAP50"]} | {"mAP50-95": stats.get("metrics/mAP50-95(B)")}


def main():
    clean = evaluate_attack(MODEL_PATH, YAML_PATH, SPLIT, attack=None)
    clean_map50 = clean.get("metrics/mAP50(B)")
    print(f"clean mAP50 = {clean_map50:.4f}")

    for attack_name, cls in [("random_radiometric", RandomSmoothRadiometricAttack), ("aria", ARIAAttack)]:
        drops = []
        for seed in SEEDS:
            attack = cls.from_severity_name("strong")
            stats = evaluate_attack(MODEL_PATH, YAML_PATH, SPLIT, attack=attack, seed=seed)
            m = stats.get("metrics/mAP50(B)")
            drops.append(clean_map50 - m)
            print(f"[{attack_name} | strong | seed={seed}] mAP50={m:.4f}, drop={clean_map50-m:.4f}")
        print(f"  -> mean drop = {statistics.mean(drops):.4f} (std {statistics.stdev(drops):.4f}), "
              f"as % of clean = {100*statistics.mean(drops)/clean_map50:.2f}%\n")


if __name__ == "__main__":
    main()
