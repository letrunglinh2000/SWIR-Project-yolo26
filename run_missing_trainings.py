"""
Run every missing training sequentially. Skips a run when its
`weights/robust_best.pt` (SWIR-AT) or `weights/best.pt` (D0/D1) already
exists -- so this is safe to re-launch after a crash.

Each launch prints to its own log and to stdout of this driver.
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RT   = REPO / "runs" / "train"

# (base_model_pt, data_yaml, run_name, scheduler, expected_best_file, gpu_h)
JOBS = [
    # ---------- P1: NSLSR safesplit / small model D2/D3/D5 --------------
    ("yolo26s.pt", "dataset_safesplit.yaml", "yolo26s_nslsr_d2_randaug",   "d2",       "robust_best.pt", 2.5),
    ("yolo26s.pt", "dataset_safesplit.yaml", "yolo26s_nslsr_d3_snua_at",   "d3_snua",  "robust_best.pt", 2.5),
    ("yolo26s.pt", "dataset_safesplit.yaml", "yolo26s_nslsr_d5_mixed",     "d5_mixed", "robust_best.pt", 3.0),
    # ---------- P1: RASMD nano D2/D3/D5 ---------------------------------
    ("yolo26n.pt", "dataset_rasmd.yaml",    "yolo26n_rasmd_d2_randaug",    "d2",       "robust_best.pt", 1.5),
    ("yolo26n.pt", "dataset_rasmd.yaml",    "yolo26n_rasmd_d3_snua_at",    "d3_snua",  "robust_best.pt", 1.5),
    ("yolo26n.pt", "dataset_rasmd.yaml",    "yolo26n_rasmd_d5_mixed",      "d5_mixed", "robust_best.pt", 1.5),
    # ---------- P2: RASMD small D2/D3/D5 --------------------------------
    ("yolo26s.pt", "dataset_rasmd.yaml",    "yolo26s_rasmd_d2_randaug",    "d2",       "robust_best.pt", 3.0),
    ("yolo26s.pt", "dataset_rasmd.yaml",    "yolo26s_rasmd_d3_snua_at",    "d3_snua",  "robust_best.pt", 3.0),
    ("yolo26s.pt", "dataset_rasmd.yaml",    "yolo26s_rasmd_d5_mixed",      "d5_mixed", "robust_best.pt", 3.0),
    # ---------- P2: NSLSR safesplit / D6 combined-AT --------------------
    ("yolo26n.pt", "dataset_safesplit.yaml","yolo26n_nslsr_d6_combined_at","d6_combined_at","robust_best.pt", 1.5),
    # ---------- P3: NSLSR safesplit / D4 ARIA-AT ------------------------
    ("yolo26n.pt", "dataset_safesplit.yaml","yolo26n_nslsr_d4_aria",       "d4_aria",  "robust_best.pt", 1.5),
]


def already_done(run_name, expected):
    return (RT / run_name / "weights" / expected).exists()


def main():
    launched = 0
    skipped  = 0
    failed   = []
    grand_start = time.time()

    for i, (model, data, name, sched, expected, hrs_est) in enumerate(JOBS, 1):
        if already_done(name, expected):
            print(f"[{i}/{len(JOBS)}] SKIP  {name}  (weights/{expected} already present)", flush=True)
            skipped += 1
            continue

        log_path = REPO / f"train_log_{name}.txt"
        cmd = [
            sys.executable, "train_swir_robust.py",
            "--model", model,
            "--data", data,
            "--name", name,
            "--scheduler", sched,
            # inherit train_swir_robust.py defaults for the rest (epochs=100,
            # batch=16, imgsz=640, seed=0) -- matches D0/D1 exactly
        ]
        print(f"\n[{i}/{len(JOBS)}] TRAIN {name}   (~{hrs_est:.1f} GPU-h estimate)", flush=True)
        print(f"    cmd: {' '.join(cmd)}", flush=True)
        print(f"    log: {log_path.name}", flush=True)

        t0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        print(f"    -> exit {proc.returncode}   elapsed {dt/3600:.2f} h", flush=True)

        if proc.returncode != 0:
            failed.append(name)
        launched += 1

    print(f"\n===== run_missing_trainings done ====", flush=True)
    print(f"  launched: {launched}   skipped: {skipped}   failed: {len(failed)}", flush=True)
    print(f"  wall time: {(time.time()-grand_start)/3600:.2f} h", flush=True)
    if failed:
        print(f"  failed jobs (see train_log_<name>.txt):", flush=True)
        for n in failed:
            print(f"    {n}", flush=True)


if __name__ == "__main__":
    main()
