"""
Milestone-5 robust-training launcher (D2/D3/D5, spec section 21).

Usage examples:

  D2 (random SWIR augmentation, no adversarial optimization):
    python train_swir_robust.py --model yolo26n.pt --data dataset_safesplit.yaml \\
        --name yolo26n_nslsr_d2_randaug --scheduler d2 --epochs 30

  D3 (SNUA adversarial training):
    python train_swir_robust.py --model yolo26n.pt --data dataset_safesplit.yaml \\
        --name yolo26n_nslsr_d3_snua_at --scheduler d3_snua --epochs 30

  D5 (mixed SWIR adversarial training):
    python train_swir_robust.py --model yolo26n.pt --data dataset_safesplit.yaml \\
        --name yolo26n_nslsr_d5_mixed --scheduler d5_mixed --epochs 30

Must be run through run_patched.py (or with `import robustness.sandbox_patch`
already applied) -- see that module's docstring for why.
"""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import robustness.sandbox_patch  # noqa: F401  -- must precede `ultralytics` import

from ultralytics import YOLO

from robustness.attacks.combined import CombinedSNUAARIAAttack
from robustness.attacks.aria import ARIAAttack
from robustness.attacks.random_fpn import RandomFPNAttack
from robustness.attacks.radiometric import RandomSmoothRadiometricAttack
from robustness.attacks.snua import SNUAGoAttack
from robustness.defenses.adversarial_trainer import SWIRRobustDetectionTrainer
from robustness.defenses.attack_scheduler import AttackScheduler
from robustness.defenses.checkpoint_selector import attach_robust_checkpoint_selector
from robustness.severities import ARIA_SEVERITIES, SNUA_SEVERITIES

TRAIN_STEPS = 5  # cheap training-time attack steps (spec section 23: training 3-10, eval 20-50)


def _cheap(severity_table, name, steps=TRAIN_STEPS):
    return dataclasses.replace(severity_table[name], steps=steps)


def build_scheduler(name: str, adv_ratio: float) -> AttackScheduler:
    """`adv_ratio` controls the clean-vs-attack SPLIT for the 2-entry
    schedulers (D2/D3); D5's mix is fixed at spec's literal starting weights
    (0.25 each) regardless of adv_ratio, since it already has its own
    explicit 'clean' entry."""
    if name == "d2":
        return AttackScheduler(
            weights={"clean": 1 - adv_ratio, "random_fpn": adv_ratio / 2, "random_radiometric": adv_ratio / 2},
            factories={
                "random_fpn": lambda: RandomFPNAttack.from_severity_name("medium"),
                "random_radiometric": lambda: RandomSmoothRadiometricAttack.from_severity_name("medium"),
            },
        )
    if name == "d3_snua":
        return AttackScheduler(
            weights={"clean": 1 - adv_ratio, "snua_go": adv_ratio},
            factories={"snua_go": lambda: SNUAGoAttack(_cheap(SNUA_SEVERITIES, "medium"))},
        )
    if name == "d4_aria":
        return AttackScheduler(
            weights={"clean": 1 - adv_ratio, "aria": adv_ratio},
            factories={"aria": lambda: ARIAAttack(_cheap(ARIA_SEVERITIES, "medium"))},
        )
    if name == "d5_mixed":
        return AttackScheduler(
            weights={"clean": 0.25, "snua_go": 0.25, "aria": 0.25, "combined": 0.25},
            factories={
                "snua_go": lambda: SNUAGoAttack(_cheap(SNUA_SEVERITIES, "medium")),
                "aria": lambda: ARIAAttack(_cheap(ARIA_SEVERITIES, "medium")),
                "combined": lambda: CombinedSNUAARIAAttack(_cheap(SNUA_SEVERITIES, "medium"), _cheap(ARIA_SEVERITIES, "medium")),
            },
        )
    if name == "d6_combined_at":
        # Adversarial training on the combined SNUA+ARIA attack ONLY (no schedule
        # mix). Fair baseline for "does D5's 4-way schedule beat plain
        # combined-AT?" -- both use the same combined attack for the adversarial
        # half of each batch; D6 has no aria-alone or snua_go-alone entries.
        return AttackScheduler(
            weights={"clean": 1 - adv_ratio, "combined": adv_ratio},
            factories={
                "combined": lambda: CombinedSNUAARIAAttack(_cheap(SNUA_SEVERITIES, "medium"), _cheap(ARIA_SEVERITIES, "medium")),
            },
        )
    raise ValueError(f"unknown scheduler '{name}' (choices: d2, d3_snua, d4_aria, d5_mixed, d6_combined_at)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--scheduler", required=True, choices=["d2", "d3_snua", "d4_aria", "d5_mixed", "d6_combined_at"])
    p.add_argument("--adv-ratio", type=float, default=0.5)
    p.add_argument("--val-split", default="val")
    p.add_argument("--robust-lambda", type=float, default=0.5)
    # Defaults matched EXACTLY to the existing D0/D1 baselines' args.yaml
    # (yolo26n_nslsr_safesplit, yolo26n_nslsr_pgd_safesplit: epochs=100,
    # patience=50, batch=16, imgsz=640, optimizer=auto, seed=0) so the
    # defense evaluation matrix compares training RECIPE, not training
    # BUDGET -- confirmed via `Get-Content runs/train/<d0/d1>/args.yaml`.
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--project", default="runs/train")  # matches train.py/train_attack.py convention
    p.add_argument("--exist-ok", action="store_true", default=True)
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] model file '{args.model}' not found")
        return

    yaml_path = Path(args.data)
    scheduler = build_scheduler(args.scheduler, args.adv_ratio)
    SWIRRobustDetectionTrainer.attack_scheduler = scheduler
    SWIRRobustDetectionTrainer.adv_ratio = 1.0  # scheduler's own "clean" weight already encodes the mix

    print("\n==========================================")
    print(" SWIR ROBUST TRAINING (Milestone 5)")
    print("==========================================")
    print(f" Model       : {args.model}")
    print(f" Data config : {yaml_path}")
    print(f" Scheduler   : {args.scheduler}  weights={dict(zip(scheduler.names, scheduler.probs))}")
    print(f" Epochs      : {args.epochs}   Batch: {args.batch}   Device: {args.device}")
    print(f" Robust-selection lambda: {args.robust_lambda}  (val split: {args.val_split})")
    print("==========================================\n")

    model = YOLO(args.model)

    # Robust checkpoint-selection callback: fixed cheap SNUA-GO(weak, 5 steps)
    # attack used for the periodic robust-score check (section 13.3) -- a
    # DIFFERENT, cheaper config than the D3/D5 training attack itself, and
    # different again from the full 20-step evaluation attack used later in
    # the defense evaluation matrix. All three are logged separately.
    selection_attack = SNUAGoAttack(_cheap(SNUA_SEVERITIES, "weak", steps=5))
    selector = attach_robust_checkpoint_selector(
        model, yaml_path, args.val_split, selection_attack,
        lam=args.robust_lambda, batch_size=args.batch, workers=args.workers,
    )

    proj_dir = str(Path(args.project).resolve()) if args.project else None

    results = model.train(
        trainer=SWIRRobustDetectionTrainer,
        data=str(yaml_path.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        save=True,
        project=proj_dir,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/train") / args.name
    rule_path = run_dir / "robust_checkpoint_rule.json"
    selector.save_rule(rule_path)
    print(f"\nSaved robust checkpoint-selection rule/history to {rule_path}")
    print(f"Best robust-score checkpoint: epoch {selector.best_epoch}, S={selector.best_score:.4f} "
          f"-> {run_dir}/weights/robust_best.pt")


if __name__ == "__main__":
    main()
