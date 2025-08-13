
#!/usr/bin/env python
"""
CUPAC/CUPED-style variance reduction using a pre-period proxy via GBM (sklearn).
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def run_cupac_gbm(df: pd.DataFrame, y: str="conversion", g: str="treatment",
                  features: Optional[Iterable[str]]=None, *, random_state: int=7) -> pd.DataFrame:
    """
    Fit GBM to predict outcome ignoring treatment, then regress y on g and the prediction (covariate adjustment).
    Returns a one-row DataFrame with OR and CI from logistic regression with the CUPAC covariate.
    """
    feats = list(features) if features else [c for c in df.columns if c.startswith("f")]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    yb = df[y].astype("float32")
    gbm = GradientBoostingRegressor(random_state=random_state)
    gbm.fit(X, yb)
    mu_hat = gbm.predict(X)
    # Logistic regression y ~ g + mu_hat
    X2 = pd.DataFrame({"intercept": 1.0, g: df[g].astype(float), "mu_hat": mu_hat})
    model = sm.GLM(yb, X2, family=sm.families.Binomial())
    res = model.fit(cov_type="HC1")
    coef = float(res.params[g]); se = float(res.bse[g]); z = coef/se if se>0 else np.nan
    from scipy.stats import norm
    p = float(2*(1-norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    lo, hi = coef - 1.96*se, coef + 1.96*se
    out = pd.DataFrame([{
        "odds_ratio_adj": float(np.exp(coef)),
        "or_adj_ci_low": float(np.exp(lo)),
        "or_adj_ci_high": float(np.exp(hi)),
        "z": float(z),
        "pvalue": float(p),
        "r2_mu_hat": float(np.corrcoef(mu_hat, yb)[0,1]**2)
    }])
    return out

if __name__ == "__main__":
    import argparse, sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from src.data_loader import make_demo_df, load_and_validate
    p = argparse.ArgumentParser(description="CUPAC with GBM variance reduction")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", type=str, default="reports/tables/cupac_summary_gbm.csv")
    args = p.parse_args()
    df = make_demo_df(30000) if args.demo else load_and_validate(args.data)
    res = run_cupac_gbm(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res)
