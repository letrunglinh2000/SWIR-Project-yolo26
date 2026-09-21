import multiprocessing.pool
from concurrent.futures import ThreadPoolExecutor

class _SafeThreadPool:
    def __init__(self, processes=None, *a, **kw):
        self._executor = ThreadPoolExecutor(max_workers=processes or 4)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self._executor.shutdown(wait=True)
        return False
    def imap(self, func, iterable, chunksize=1):
        return self._executor.map(func, iterable)
    def imap_unordered(self, func, iterable, chunksize=1):
        return self._executor.map(func, iterable)
    def map(self, func, iterable, chunksize=1):
        return list(self._executor.map(func, iterable))
    def close(self):
        pass
    def join(self):
        self._executor.shutdown(wait=True)
    def terminate(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

multiprocessing.pool.ThreadPool = _SafeThreadPool


import sys
from pathlib import Path
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_DICT

BASE = Path(r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26")
YAML_PATH = BASE / "dataset_rasmd.yaml"
DEVICE = "0"
IMGSZ = 640
BATCH_SIZE = 8
WORKERS = 0

PGD_EPS = 8.0/255.0
PGD_ALPHA = 2.0/255.0
PGD_STEPS = 20

MODELS = {
    "yolo26n_rasmd_clean": BASE/"runs/train/yolo26n_rasmd/weights/best.pt",
    "yolo26s_rasmd_clean": BASE/"runs/train/yolo26s_rasmd/weights/best.pt",
    "yolo26n_rasmd_pgd":   BASE/"runs/train/yolo26n_rasmd_pgd/weights/best.pt",
    "yolo26s_rasmd_pgd":   BASE/"runs/train/yolo26s_rasmd_pgd/weights/best.pt",
}

def pgd_attack(model, batch, eps, alpha, steps, random_start=True):
    x = batch["img"].detach()
    if random_start:
        delta = torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
    else:
        x_adv = x.clone()
    x_adv = x_adv.detach()

    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    bn_states = [m.training for m in bn_layers]
    orig_state = model.training
    model.train()
    for m in bn_layers:
        m.eval()
    try:
        for _ in range(steps):
            x_adv.requires_grad_(True)
            adv_batch = batch.copy()
            adv_batch["img"] = x_adv
            output = model(adv_batch)
            loss = output[0] if isinstance(output, (tuple, list)) else output
            attack_loss = loss.sum()
            grad = torch.autograd.grad(outputs=attack_loss, inputs=x_adv,
                                        retain_graph=False, create_graph=False, only_inputs=True)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            x_adv = torch.clamp(x + delta, min=0.0, max=1.0).detach()
    finally:
        for m, s in zip(bn_layers, bn_states):
            m.train(s)
        model.train(orig_state)
    return x_adv


def build_validator(model_path, yaml_path, device):
    yolo = YOLO(str(model_path))
    detector = yolo.model.to(device).float()

    if isinstance(detector.args, dict):
        saved_args = {k: v for k, v in detector.args.items() if k in DEFAULT_CFG_DICT}
        detector.args = get_cfg(overrides=saved_args)
    detector.criterion = None
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    validator = DetectionValidator(args={
        "model": str(model_path), "data": str(yaml_path.resolve()), "split": "test",
        "imgsz": IMGSZ, "batch": BATCH_SIZE, "device": DEVICE, "workers": WORKERS,
        "rect": True, "conf": 0.001, "iou": 0.7,
        "plots": False, "save_json": False, "save_txt": False, "verbose": False, "mode": "val",
    })
    validator.training = False
    validator.device = device
    validator.args.quantize = None
    validator.data = check_det_dataset(str(yaml_path.resolve()), split="test")
    validator.stride = int(detector.stride.max().item())
    validator.dataloader = validator.get_dataloader(validator.data["test"], BATCH_SIZE)
    validator.init_metrics(detector)
    return detector, validator


def run_eval(model_path, yaml_path, attack):
    device = select_device(DEVICE)
    detector, validator = build_validator(model_path, yaml_path, device)

    for batch in validator.dataloader:
        batch = validator.preprocess(batch)
        if attack:
            adv_img = pgd_attack(detector, batch, PGD_EPS, PGD_ALPHA, PGD_STEPS)
            batch["img"] = adv_img
        detector.eval()
        with torch.inference_mode():
            preds = detector(batch["img"])
            preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)

    validator.gather_stats()
    stats = validator.get_stats()
    validator.finalize_metrics()
    return stats


if __name__ == "__main__":
    import json
    results = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"SKIP missing: {path}")
            continue
        for cond in ["clean", "pgd20"]:
            print(f"=== {name} | {cond} ===", flush=True)
            stats = run_eval(path, YAML_PATH, attack=(cond == "pgd20"))
            row = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
            results[f"{name}__{cond}"] = row
            print(name, cond, {k: round(v,4) for k,v in row.items() if 'map' in k.lower() or 'precision' in k.lower() or 'recall' in k.lower()}, flush=True)

    with open(BASE / "pgd_robustness_grid_rasmd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")
