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


import json
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"D:\letrunglinh\1.Paper_IVCL\11.SWIR_Project\3.test\yolo26")
YAML_PATH = BASE / "dataset_rasmd.yaml"

MODELS = {
    "yolo26n_clean": BASE/"runs/train/yolo26n_rasmd/weights/best.pt",
    "yolo26s_clean": BASE/"runs/train/yolo26s_rasmd/weights/best.pt",
    "yolo26n_pgd":   BASE/"runs/train/yolo26n_rasmd_pgd/weights/best.pt",
    "yolo26s_pgd":   BASE/"runs/train/yolo26s_rasmd_pgd/weights/best.pt",
}

names = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']

results = {}
for tag, path in MODELS.items():
    m = YOLO(str(path))
    r = m.val(data=str(YAML_PATH), split="val", workers=0, batch=8, imgsz=640, plots=False, verbose=False)
    per_class_ap50 = {}
    ap_class_idx = list(r.box.ap_class_index)
    ap50_per_class = r.box.ap50  # array aligned with ap_class_index
    for idx, cid in enumerate(ap_class_idx):
        per_class_ap50[names[int(cid)]] = float(ap50_per_class[idx])
    results[tag] = {
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "per_class_ap50": per_class_ap50,
    }
    print(tag, "mAP50=", round(results[tag]["mAP50"],3), "per_class:", {k: round(v,3) for k,v in per_class_ap50.items()})

with open(BASE / "rasmd_perclass_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE")
