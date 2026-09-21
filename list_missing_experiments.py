"""
Enumerate the experiments implied by the codebase and mark which ones
have NOT been run yet. Output: missing_experiments.csv.

Grouped by category (training vs eval vs diagnostic) rather than a full
Cartesian grid, because the full grid is 500+ cells and most of it isn't
what a paper needs. Each row is a coherent unit of work with a rough
GPU-time estimate and a note on WHY it matters for the paper.

Cost estimates are rough (RTX 3090 Ti):
  - Training:   ~1.0-1.5 h per 100-epoch nano run on NSLSR safesplit;
                ~2-3 h per 100-epoch small run
  - Eval pass:  ~30-90 s per (checkpoint, attack, seed) on NSLSR val/test
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent

# ---- What exists on disk ------------------------------------------------
RUN_DIRS = {p.name for p in (REPO / "runs" / "train").iterdir() if p.is_dir()}

# ---- What has been measured --------------------------------------------
measured = defaultdict(set)   # (dataset, split_variant, model, training) -> {attacks}
for r in csv.DictReader(open(REPO / "results_summary_long.csv")):
    measured[(r["dataset"], r["split_variant"], r["model"], r["training"])].add(r["attack"])

SWIR_ATTACKS_STD = {"snua_go_weak", "snua_go_medium", "aria_medium", "combined_medium"}

rows = []

# ==== A. TRAINING EXPERIMENTS ============================================
# only the paper-relevant missing runs (nano/small x standard/pgd_at/D2/D3/D4/D5
# x NSLSR-safesplit/RASMD). The official-leaky split is deliberately excluded.
trainings_missing = [
    # dataset,           model,    training,           run_dir_expected,             prio, hrs, why
    ("NSLSR-safesplit", "small",  "random_swir_aug",  "yolo26s_nslsr_d2_randaug",   "P1", 2.5,
     "matched random control for the small model; lets you show the D3/D5 win is not a model-size artifact"),
    ("NSLSR-safesplit", "small",  "snua_at",          "yolo26s_nslsr_d3_snua_at",   "P1", 2.5,
     "does the SWIR-AT gain hold when moving from nano to small? Currently only nano is tested"),
    ("NSLSR-safesplit", "small",  "mixed_swir_at",    "yolo26s_nslsr_d5_mixed",     "P1", 3.0,
     "D5 is the papers best defense -- needs a size ablation. Currently only nano"),
    ("NSLSR-safesplit", "nano",   "aria_at",          "yolo26n_nslsr_d4_aria",      "P3", 1.5,
     "D4 branch is coded (build_scheduler) but never launched; only needed if ARIA becomes a real threat at strong severity"),
    ("NSLSR-safesplit", "nano",   "combined_at",      "yolo26n_nslsr_d6_combined_at", "P2", 1.5,
     "no D6 in the codebase yet -- adversarial training with the combined attack alone. Distinct from D5s mixed schedule; a fair baseline for whether the mixed schedule is better than pure combined-AT"),
    ("RASMD",           "nano",   "random_swir_aug",  "yolo26n_rasmd_d2_randaug",   "P1", 1.5,
     "RASMD has ZERO SWIR-domain defenses. Without D2 baseline, no RASMD SWIR robustness claim can be made"),
    ("RASMD",           "nano",   "snua_at",          "yolo26n_rasmd_d3_snua_at",   "P1", 1.5,
     "does SNUA-AT transfer to the 6-class RASMD? Only NSLSR-safesplit has it"),
    ("RASMD",           "nano",   "mixed_swir_at",    "yolo26n_rasmd_d5_mixed",     "P1", 1.5,
     "external validity of D5 -- second dataset, six classes vs one, different sensor -- confirms/rebuts the finding"),
    ("RASMD",           "small",  "random_swir_aug",  "yolo26s_rasmd_d2_randaug",   "P2", 3.0,
     "small-model + RASMD SWIR baseline"),
    ("RASMD",           "small",  "snua_at",          "yolo26s_rasmd_d3_snua_at",   "P2", 3.0,
     "cross-dataset AND cross-model SWIR-AT check"),
    ("RASMD",           "small",  "mixed_swir_at",    "yolo26s_rasmd_d5_mixed",     "P2", 3.0,
     "the strongest cross-dataset generalisation test for D5"),
]
for ds, model, tr, rd, prio, hrs, why in trainings_missing:
    rows.append({
        "category": "A_training",
        "id": f"TR-{ds.split('-')[0]}-{model}-{tr}",
        "dataset_split": ds, "model": model, "training": tr, "attack": "",
        "status": "MISSING (no run dir)" if rd not in RUN_DIRS else f"run dir '{rd}' exists -- may already be done",
        "priority": prio, "gpu_hours_estimate": hrs,
        "why": why,
    })

# ==== B. EVALUATION EXPERIMENTS ==========================================
# Attack a checkpoint that ALREADY EXISTS with a condition never measured.

# B1: SWIR attacks vs existing RASMD checkpoints -------------------------
for model in ("nano", "small"):
    for training in ("standard", "adversarial"):
        run_key = f"yolo26{model[0]}_rasmd" + ("_pgd" if training == "adversarial" else "")
        for attack in sorted(SWIR_ATTACKS_STD):
            key = ("RASMD", "stratified (own)", model, training)
            if attack not in measured[key]:
                rows.append({
                    "category": "B_eval",
                    "id": f"EV-RASMD-{model}-{training}-{attack}",
                    "dataset_split": "RASMD", "model": model, "training": training, "attack": attack,
                    "status": f"MISSING (checkpoint {run_key} exists, never attacked with SWIR)",
                    "priority": "P1", "gpu_hours_estimate": 0.04,
                    "why": "RASMD has never been probed with any SWIR-domain attack -- the whole SWIR-attack paper claim is single-dataset without this",
                })

# B2: SWIR attacks vs existing yolo26s NSLSR-safesplit checkpoints -------
for training in ("standard", "adversarial"):
    for attack in sorted(SWIR_ATTACKS_STD):
        key = ("NSLSR", "leakage-safe (segment)", "small", training)
        if attack not in measured[key]:
            rows.append({
                "category": "B_eval",
                "id": f"EV-NSLSRsafe-small-{training}-{attack}",
                "dataset_split": "NSLSR-safesplit", "model": "small", "training": training, "attack": attack,
                "status": "MISSING (checkpoint exists)",
                "priority": "P2", "gpu_hours_estimate": 0.05,
                "why": "small-model SWIR-attack numbers exist for NO training recipe; nano-only argument is weaker",
            })

# B3: matched random controls against every OTHER checkpoint (only D0 has them)
for defense in ("D1_pgd_at", "D2_random_swir_aug", "D3_snua_at", "D5_mixed_swir_at"):
    for atk in ("random_fpn_medium", "random_radiometric_medium"):
        rows.append({
            "category": "B_eval",
            "id": f"EV-NSLSRsafe-nano-{defense}-{atk}",
            "dataset_split": "NSLSR-safesplit", "model": "nano",
            "training": defense.split("_", 1)[1], "attack": atk,
            "status": "MISSING (only D0 has matched random controls)",
            "priority": "P2", "gpu_hours_estimate": 0.04,
            "why": "control condition for each defense -- is the defense doing better than seeing the random-perturbation version? Currently answerable only for standard training",
        })

# B4: strong severities never measured on any defense
for defense in ("standard", "pgd_at", "random_swir_aug", "snua_at", "mixed_swir_at"):
    for atk in ("snua_go_strong", "aria_strong", "combined_strong"):
        rows.append({
            "category": "B_eval",
            "id": f"EV-NSLSRsafe-nano-{defense}-{atk}",
            "dataset_split": "NSLSR-safesplit", "model": "nano", "training": defense, "attack": atk,
            "status": "MISSING (strong severity never measured)",
            "priority": "P3", "gpu_hours_estimate": 0.04,
            "why": "shows the failure surface -- how quickly does robustness degrade as budget grows? Currently the medium point is the only data point per attack",
        })

# ==== C. DIAGNOSTIC / OUTSTANDING SANITY-CHECK SCRIPTS ===================
diag = [
    ("aria_strong_check",
     "ARIA at strong severity (eps_field=0.20) vs random_radiometric strong",
     "P1", 0.05,
     "script exists (robustness/experiments/aria_strong_check.py) but no log/results on disk. Needed BEFORE ARIA can be reported as either a real attack or a paper limitation"),
    ("SNUA_vs_random_on_defenses",
     "SNUA vs random-FPN comparison run against D1/D2/D3/D5, not just D0",
     "P2", 0.2,
     "currently the 'is it optimization or just augmentation?' argument holds for D0 only; if it fails on the AT defenses, the interpretation of D3 changes"),
    ("test_set_bootstrap_CIs",
     "Per-image bootstrap of mAP50 on the 144-image test split",
     "P2", 0.3,
     "seed-std is 0.001-0.014 mAP50, but that under-states real uncertainty because the split is only 144 images. Bootstrap CIs are what a reviewer will ask for on a small test set"),
    ("val_test_composition_audit",
     "Investigate why test mAP50 is systematically 0.24-0.35 HIGHER than val on identical checkpoints",
     "P1", 0.5,
     "found in the defense-matrix val->test delta -- suggests val split is systematically harder. If real, every val number in the aggregate table is pessimistic; if a split bug, may need re-splitting and re-running"),
    ("class_balance_check_NSLSR",
     "Small-object vs large-object ship recall breakdown on val vs test",
     "P2", 0.2,
     "if val is harder because it over-samples small ships, that CAUSES the val->test gap and explains the finding; if not, the gap is more concerning"),
]
for name, desc, prio, hrs, why in diag:
    rows.append({
        "category": "C_diagnostic",
        "id": f"DIAG-{name}",
        "dataset_split": "", "model": "", "training": "", "attack": "",
        "status": f"NOT RUN -- {desc}",
        "priority": prio, "gpu_hours_estimate": hrs,
        "why": why,
    })

# ---- write --------------------------------------------------------------
fields = ["category", "id", "dataset_split", "model", "training", "attack",
          "status", "priority", "gpu_hours_estimate", "why"]
out = REPO / "missing_experiments.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"wrote {out.name}  ({len(rows)} missing/pending experiments)")

print("\nBy priority:")
for p, n in sorted(Counter(r["priority"] for r in rows).items()):
    hrs = sum(r["gpu_hours_estimate"] for r in rows if r["priority"] == p)
    print(f"  {p}: {n:3d}  (~{hrs:.1f} GPU-h)")

print("\nBy category:")
for c, n in sorted(Counter(r["category"] for r in rows).items()):
    hrs = sum(r["gpu_hours_estimate"] for r in rows if r["category"] == c)
    print(f"  {c}: {n:3d}  (~{hrs:.1f} GPU-h)")

print(f"\nTotal (everything): ~{sum(r['gpu_hours_estimate'] for r in rows):.1f} GPU-h")
print(f"P1 only (highest-value paper gaps): ~{sum(r['gpu_hours_estimate'] for r in rows if r['priority']=='P1'):.1f} GPU-h")
