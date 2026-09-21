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

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_DICT
from pathlib import Path

BASE = Path(r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26")
yaml_path = BASE / "dataset_safesplit.yaml"
model_path = BASE / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"

device = select_device("0")
yolo = YOLO(str(model_path))
detector = yolo.model.to(device).float()
if isinstance(detector.args, dict):
    saved_args = {k: v for k, v in detector.args.items() if k in DEFAULT_CFG_DICT}
    detector.args = get_cfg(overrides=saved_args)
print("detector.args type after fix:", type(detector.args))
print("hyp.box present:", hasattr(detector.args, "box"), getattr(detector.args, "box", None))
detector.eval()

validator = DetectionValidator(args={
    "model": str(model_path), "data": str(yaml_path.resolve()), "split": "val",
    "imgsz": 640, "batch": 4, "device": "0", "workers": 0,
    "rect": True, "conf": 0.001, "iou": 0.7,
    "plots": False, "save_json": False, "save_txt": False, "verbose": False, "mode": "val",
})
validator.device = device
validator.data = check_det_dataset(str(yaml_path.resolve()), split="val")
validator.stride = int(detector.stride.max().item())
validator.dataloader = validator.get_dataloader(validator.data["val"], 4)

batch = next(iter(validator.dataloader))
raw_img = batch["img"]
print("RAW batch img dtype", raw_img.dtype, "min", raw_img.min().item(), "max", raw_img.max().item())
pp = validator.preprocess(batch)
print("PREPROCESSED img dtype", pp["img"].dtype, "min", pp["img"].min().item(), "max", pp["img"].max().item(), "shape", pp["img"].shape)
print("requires_grad after preprocess:", pp["img"].requires_grad)

# Confirm loss is differentiable pre-NMS: check what model(batch) returns in train-mode-for-loss vs eval mode
detector.train()
for m in detector.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
img = pp["img"].clone().requires_grad_(True)
batch2 = dict(pp)
batch2["img"] = img
out = detector(batch2)
print("output type:", type(out), "is tuple/list:", isinstance(out, (tuple, list)))
if isinstance(out, (tuple, list)):
    print("len:", len(out), "first elem type:", type(out[0]), "shape:", getattr(out[0], "shape", None))
loss = out[0] if isinstance(out, (tuple, list)) else out
total = loss.sum()
total.backward()
print("grad reaches image:", img.grad is not None, "grad abs max:", img.grad.abs().max().item() if img.grad is not None else None)
