"""Tabla resumen de optimal_alphas.json (modo all y divided) con exactitud balanceada en test."""
import json
from pathlib import Path
import pandas as pd

import sys
ATT = len(sys.argv) > 1 and sys.argv[1] == "atencion"
BASE = Path(__file__).resolve().parent.parent.parent / ("results/voting_predictions" if ATT else "results/voting_predictions/residual")
rows = []
for mode, path in (("all", BASE / "optimal_alphas.json"), ("divided", BASE / "divided/optimal_alphas.json")):
    if not path.exists():
        continue
    for model, methods in json.load(open(path)).items():
        for method, r in methods.items():
            t, b = r.get("test_metrics"), r.get("baseline_test_metrics")
            if not t:
                continue
            rows.append({"tareas": mode, "modelo": model, "metodo": method, "alpha": r["best_alpha"],
                         "bal_acc_base": b["balanced_accuracy"], "bal_acc_steer": t["balanced_accuracy"],
                         "acc_base": b["accuracy"], "acc_steer": t["accuracy"],
                         "recall_contra_base": b["recall_en_contra"], "recall_contra_steer": t["recall_en_contra"],
                         "pred_a_favor_steer": t["pred_a_favor_rate"]})
df = pd.DataFrame(rows).round(1)
out = (BASE if ATT else BASE.parent) / ("resumen_atencion.csv" if ATT else "resumen_residual.csv")
df.to_csv(out, index=False)
print(df.to_string(index=False))
print(f"-> {out}")
