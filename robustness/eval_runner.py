"""
Evaluation runner: wires a YOLO26 checkpoint + a dataset split + a
`BaseAttack` instance into a standard P/R/mAP50/mAP50-95 result, using the
exact same `DetectionValidator` construction (rect batching, conf=0.001,
iou=0.7, plots/json/txt disabled) already validated across every prior
result in this project's aggregate results table -- so numbers produced
through this new common-attack-framework path are directly comparable to
everything already measured with the ad hoc per-script evaluators.

IMPORTANT: import `robustness.sandbox_patch` before importing this module in
any entry-point script (see that module's docstring).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import select_device

from .core import BaseAttack, prepare_detector_for_attack


def build_validator_for_detector(
    detector,
    yaml_path: Path,
    split: str,
    model_path_str: str = "in_memory",
    device_str: str = "0",
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 0,
):
    """Same DetectionValidator construction as `build_validator`, but for an
    ALREADY-LOADED detector (e.g. a live training run's `trainer.model`) --
    no `YOLO(str(model_path))` reload from disk. Used by the periodic
    robust-checkpoint-selection callback (robustness/defenses/checkpoint_selector.py)
    so the per-epoch robustness check costs one extra val-split pass, not a
    disk round-trip. Does NOT mutate the caller's detector mode/requires_grad
    -- caller is responsible for that (a training-time caller must restore
    `.train()` + `requires_grad_(True)` afterward)."""
    device = select_device(device_str)
    validator = DetectionValidator(args={
        "model": model_path_str, "data": str(yaml_path.resolve()), "split": split,
        "imgsz": imgsz, "batch": batch_size, "device": device_str, "workers": workers,
        "rect": True, "conf": 0.001, "iou": 0.7,
        "plots": False, "save_json": False, "save_txt": False, "verbose": False, "mode": "val",
    })
    validator.training = False
    validator.device = device
    validator.args.quantize = None
    validator.data = check_det_dataset(str(yaml_path.resolve()), split=split)
    validator.stride = int(detector.stride.max().item())
    validator.dataloader = validator.get_dataloader(validator.data[split], batch_size)
    validator.init_metrics(detector)
    return validator


def build_validator(
    model_path: Path,
    yaml_path: Path,
    split: str,
    device_str: str = "0",
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 0,
):
    device = select_device(device_str)
    yolo = YOLO(str(model_path))
    detector = yolo.model.to(device).float()
    prepare_detector_for_attack(detector)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    validator = build_validator_for_detector(
        detector, yaml_path, split, model_path_str=str(model_path),
        device_str=device_str, imgsz=imgsz, batch_size=batch_size, workers=workers,
    )
    return detector, validator


def run_eval_loop(detector, validator, attack: Optional[BaseAttack], seed: Optional[int] = None) -> Dict[str, Any]:
    """Shared eval loop: one full pass over `validator.dataloader`, optionally
    attacking every batch first. Caller owns detector/validator lifecycle
    (mode, requires_grad, device) -- this function restores `detector.eval()`
    before each inference call but does not otherwise touch training state,
    so it is safe to call from inside a training loop (checkpoint_selector.py)
    as well as from a standalone evaluation script."""
    for batch in validator.dataloader:
        batch = validator.preprocess(batch)
        if attack is not None:
            result = attack.generate(detector, batch, seed=seed)
            batch = dict(batch)
            batch["img"] = result.adv_images
        detector.eval()
        with torch.inference_mode():
            preds = detector(batch["img"])
            preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)

    validator.gather_stats()
    stats = validator.get_stats()
    validator.finalize_metrics()
    return stats


def evaluate_attack(
    model_path: Path,
    yaml_path: Path,
    split: str,
    attack: Optional[BaseAttack],
    device_str: str = "0",
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one full pass over `split`, optionally attacking every batch with
    `attack` first (pass `attack=None` for a clean evaluation), and return
    the ultralytics stats dict (same keys as every existing result in this
    project: 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
    'metrics/mAP50-95(B)', plus per-class breakdowns when nc>1).
    """
    model_path = Path(model_path)
    yaml_path = Path(yaml_path)
    detector, validator = build_validator(model_path, yaml_path, split, device_str, imgsz, batch_size, workers)
    return run_eval_loop(detector, validator, attack, seed=seed)
