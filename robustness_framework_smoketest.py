import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import robustness.sandbox_patch  # noqa: F401  (must precede ultralytics import)

from robustness.attacks.pgd import PGDAttack
from robustness.eval_runner import evaluate_attack

BASE = Path(r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26")
model_path = BASE / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
yaml_path = BASE / "dataset_safesplit.yaml"

print("=== clean (PGDAttack steps=0, identity path) ===")
stats_clean = evaluate_attack(model_path, yaml_path, "test", attack=PGDAttack(steps=0))
print({k: round(v, 4) for k, v in stats_clean.items() if isinstance(v, (int, float)) and ("map" in k.lower() or "precision" in k.lower() or "recall" in k.lower())})

print("=== PGD-20 (eps=8/255, alpha=2/255, steps=20, random_start=True) ===")
stats_pgd = evaluate_attack(model_path, yaml_path, "test", attack=PGDAttack(eps=8/255, alpha=2/255, steps=20, random_start=True), seed=0)
print({k: round(v, 4) for k, v in stats_pgd.items() if isinstance(v, (int, float)) and ("map" in k.lower() or "precision" in k.lower() or "recall" in k.lower())})

print("=== truly clean (attack=None) ===")
stats_none = evaluate_attack(model_path, yaml_path, "test", attack=None)
print({k: round(v, 4) for k, v in stats_none.items() if isinstance(v, (int, float)) and ("map" in k.lower() or "precision" in k.lower() or "recall" in k.lower())})
