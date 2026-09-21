"""
M1 regression test: the ported PGDAttack (common framework) must reproduce
the original ad hoc PGD script's clean and PGD-20 numbers on the NSLSR
leakage-safe test split within a documented tolerance.

Anchor source: pgd_robustness_grid_safesplit_full_results.json, key
'yolo26n_nslsr_safesplit__best__clean' / '...__pgd20' (produced this
session by the original per-script PGD implementation, BEFORE this
framework existed).

Run directly with `python robustness/tests/test_pgd_regression.py` (no
pytest dependency assumed -- plain asserts + a PASS/FAIL summary so this
runs identically in any environment that has ultralytics/torch).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401  (must precede ultralytics import)

from robustness.attacks.pgd import PGDAttack
from robustness.eval_runner import evaluate_attack

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"

# Anchor numbers (original per-script PGD implementation, same checkpoint/split)
ANCHOR_CLEAN = {"mAP50": 0.9311, "mAP50-95": 0.6483, "precision": 0.8697, "recall": 0.8384}
ANCHOR_PGD20 = {"mAP50": 0.0, "mAP50-95": 0.0}  # precision/recall not gated -- both near-zero and
                                                  # stochastic (random-start dependent) by construction

CLEAN_TOL = 1e-3     # clean eval is deterministic -- must match almost exactly
PGD_MAP_TOL = 0.05   # PGD-20 has random-start stochasticity; both anchor and this run land
                      # in a near-total-collapse regime (mAP50 < 0.01), so the tolerance only
                      # needs to confirm "still collapsed", not bit-exactness


def _extract(stats):
    return {
        "mAP50": stats.get("metrics/mAP50(B)"),
        "mAP50-95": stats.get("metrics/mAP50-95(B)"),
        "precision": stats.get("metrics/precision(B)"),
        "recall": stats.get("metrics/recall(B)"),
    }


def main():
    failures = []

    print("=== regression: clean (attack=None) ===")
    clean = _extract(evaluate_attack(MODEL_PATH, YAML_PATH, "test", attack=None))
    print(clean)
    for k, anchor_v in ANCHOR_CLEAN.items():
        diff = abs(clean[k] - anchor_v)
        ok = diff <= CLEAN_TOL
        print(f"  {k}: got {clean[k]:.4f} vs anchor {anchor_v:.4f} (diff {diff:.4f}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"clean.{k}")

    print("=== regression: PGD-20 (eps=8/255, alpha=2/255, steps=20, random_start=True) ===")
    pgd = _extract(evaluate_attack(
        MODEL_PATH, YAML_PATH, "test",
        attack=PGDAttack(eps=8 / 255, alpha=2 / 255, steps=20, random_start=True),
        seed=0,
    ))
    print(pgd)
    for k, anchor_v in ANCHOR_PGD20.items():
        diff = abs(pgd[k] - anchor_v)
        ok = diff <= PGD_MAP_TOL
        print(f"  {k}: got {pgd[k]:.4f} vs anchor {anchor_v:.4f} (diff {diff:.4f}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"pgd20.{k}")

    print()
    if failures:
        print(f"REGRESSION TEST: FAIL ({len(failures)} check(s) failed: {failures})")
        sys.exit(1)
    else:
        print("REGRESSION TEST: ALL PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
