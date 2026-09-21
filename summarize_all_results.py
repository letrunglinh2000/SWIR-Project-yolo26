"""
Build two summary CSVs from aggregate_results_table3.csv:

  1) results_summary_long.csv -- tidy long form, one row per experiment,
     mean +- std over seeds, sorted by dataset/split/model/training/attack.
     Useful for scripts / plotting / spot-checking any single cell.

  2) results_summary_pivot_mAP50.csv -- wide pivot, one row per
     (dataset, split_variant, model, training, checkpoint, eval_split),
     one column per attack. Cell = mean mAP50. This is the paper-shaped
     view: "what fraction of clean performance does each defense keep
     under each attack?"

Also prints a headline table to the terminal.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC  = REPO / "aggregate_results_table3.csv"

def _f(v):
    return float(v) if v not in ("", None) else None

rows = []
with open(SRC, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# ---- 1) LONG form (already close to what table3 is) --------------------
long_fields = [
    "dataset", "split_variant", "model", "training", "checkpoint", "eval_split",
    "attack", "n_seeds", "mAP50", "mAP50_std", "mAP50-95", "mAP50-95_std",
    "precision", "recall", "selection_bias", "source",
]
out_long = REPO / "results_summary_long.csv"
with open(out_long, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=long_fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in long_fields})
print(f"wrote {out_long.name}  ({len(rows)} rows)")

# ---- 2) PIVOT form: mAP50 mean per attack column -----------------------
ATTACKS = ["clean", "pgd20", "snua_go_weak", "snua_go_medium",
           "aria_medium", "combined_medium"]
KEY = ("dataset", "split_variant", "model", "training", "checkpoint", "eval_split")

grid = defaultdict(dict)
for r in rows:
    k = tuple(r[c] for c in KEY)
    a = r["attack"]
    if a in ATTACKS:
        grid[k][a] = _f(r["mAP50"])

def _sort_key(k):
    ds  = {"NSLSR": 0, "RASMD": 1}.get(k[0], 2)
    sp  = {"official (leaky)": 0, "leakage-safe (segment)": 1, "stratified (own)": 2}.get(k[1], 3)
    md  = {"nano": 0, "small": 1}.get(k[2], 2)
    tr  = {"standard": 0, "adversarial": 1, "pgd_at": 2, "random_swir_aug": 3,
           "snua_at": 4, "mixed_swir_at": 5}.get(k[3], 9)
    ck  = {"best": 0, "last": 1, "robust_best": 2}.get(k[4], 9)
    es  = {"val": 0, "test": 1}.get(k[5], 2)
    return (ds, sp, md, tr, ck, es)

out_pivot = REPO / "results_summary_pivot_mAP50.csv"
with open(out_pivot, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(list(KEY) + ATTACKS + ["worst_attack", "worst_mAP50", "clean_cost_vs_D0_same_split"])

    # for "clean cost vs D0 same split" comparison
    d0_clean_by_split = {}
    for k, cells in grid.items():
        if k[3] == "standard" and k[4] == "best" and "clean" in cells:
            d0_clean_by_split.setdefault((k[0], k[1], k[2], k[5]), cells["clean"])

    for k in sorted(grid.keys(), key=_sort_key):
        cells = grid[k]
        vals = [cells.get(a) for a in ATTACKS]
        # worst-case: min over the non-clean attacks that ran for this row
        atk_vals = [(a, cells[a]) for a in ATTACKS[1:] if a in cells]
        worst_atk, worst_val = ("", "")
        if atk_vals:
            worst_atk, worst_val = min(atk_vals, key=lambda x: x[1])
        d0c = d0_clean_by_split.get((k[0], k[1], k[2], k[5]))
        clean_cost = ""
        if d0c is not None and "clean" in cells:
            clean_cost = f"{cells['clean'] - d0c:+.4f}"
        w.writerow(list(k)
                   + [f"{v:.4f}" if v is not None else "" for v in vals]
                   + [worst_atk, f"{worst_val:.4f}" if worst_val != "" else "", clean_cost])
print(f"wrote {out_pivot.name}  ({len(grid)} experiments)")

# ---- 3) headline terminal print ----------------------------------------
print("\n=== HEADLINE: NSLSR leakage-safe (segment), nano, TEST split, mAP50 ===\n")
hdr = ["training", "checkpoint"] + ATTACKS + ["worst"]
print("  " + "  ".join(f"{h:>16}" for h in hdr))
for k in sorted(grid.keys(), key=_sort_key):
    if k[:3] != ("NSLSR", "leakage-safe (segment)", "nano") or k[5] != "test":
        continue
    cells = grid[k]
    atk_vals = [cells[a] for a in ATTACKS[1:] if a in cells]
    worst = min(atk_vals) if atk_vals else float("nan")
    line = [k[3], k[4]] + [f"{cells.get(a, float('nan')):.3f}" for a in ATTACKS] + [f"{worst:.3f}"]
    print("  " + "  ".join(f"{c:>16}" for c in line))
