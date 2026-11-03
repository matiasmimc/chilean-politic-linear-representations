"""IC bootstrap (remuestreo de diputados) de la exactitud balanceada en test: baseline vs cada metodo.
Uso: uv run python scripts/steering/bootstrap_controles.py [all|divided]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ATT = len(sys.argv) > 2 and sys.argv[2] == "atencion"
BASE = Path(__file__).resolve().parent.parent.parent / ("results/voting_predictions" if ATT else "results/voting_predictions/residual")
mode = sys.argv[1] if len(sys.argv) > 1 else "divided"
d = BASE / ("divided" if mode == "divided" else "")
rng = np.random.RandomState(0)


def bacc(df):
    r = [(df[df.voto_real == c].voto_predicho == c).mean() for c in ("A FAVOR", "EN CONTRA") if (df.voto_real == c).any()]
    return float(np.mean(r)) * 100


def load(path):
    df = pd.read_parquet(path)
    test = df[df.split == "test"]
    steer = test[test.alpha > 0]
    base = test[test.alpha == 0]
    return base, (steer if len(steer) else base)


rows = []
for f in sorted(d.glob("voting_predictions_*.parquet")):
    stem = f.stem.replace("voting_predictions_", "")
    key = "_cpca" if "_cpca" in stem else "_random"
    model, method = stem.split(key)[0], key[1:] + stem.split(key, 1)[1]
    base, steer = load(f)
    mps = base.mp_uid.unique()
    b_g, s_g = base.groupby("mp_uid"), steer.groupby("mp_uid")
    deltas = []
    for _ in range(500):
        sample = rng.choice(mps, len(mps))
        b = pd.concat([b_g.get_group(m) for m in sample]); s = pd.concat([s_g.get_group(m) for m in sample if m in s_g.groups])
        deltas.append(bacc(s) - bacc(b))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    rows.append({"modelo": model, "metodo": method, "bal_acc_base": bacc(base), "bal_acc_steer": bacc(steer),
                 "delta": bacc(steer) - bacc(base), "ic95_lo": lo, "ic95_hi": hi})
out = pd.DataFrame(rows).round(2)
print(out.to_string(index=False))
out.to_csv((BASE if ATT else BASE.parent) / f"bootstrap_{mode}{'_atencion' if ATT else ''}.csv", index=False)
