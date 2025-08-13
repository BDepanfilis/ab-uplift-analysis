#!/usr/bin/env python
"""
AB stats: two-proportion z-test, SMD balance, logistic primary effect with robust SEs,
SRM check, and a CSV summary including CIs for absolute difference and OR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# ---------- Two-proportion z-test ----------
@dataclass
class TwoPropResult:
    p_control: float
    p_treat: float
    diff: float
    z: float
    pvalue: float

def two_prop_test(df: pd.DataFrame, y: str="conversion", g: str="treatment") -> TwoPropResult:
    x0 = df.loc[df[g]==0, y].to_numpy()
    x1 = df.loc[df[g]==1, y].to_numpy()
    count = np.array([x1.sum(), x0.sum()], dtype=float)
    nobs  = np.array([x1.size, x0.size], dtype=float)
    z, p = proportions_ztest(count, nobs)
    return TwoPropResult(float(x0.mean()), float(x1.mean()), float(x1.mean()-x0.mean()), float(z), float(p))

# ---------- SMD balance ----------
def smd_table(df: pd.DataFrame, cols: Iterable[str], g: str="treatment") -> pd.DataFrame:
    rows = []
    g1 = df[g]==1
    g0 = df[g]==0
    for c in cols:
        x1 = pd.to_numeric(df.loc[g1, c], errors="coerce")
        x0 = pd.to_numeric(df.loc[g0, c], errors="coerce")
        m1, m0 = x1.mean(), x0.mean()
        v1, v0 = x1.var(ddof=1), x0.var(ddof=1)
        s = np.sqrt(0.5*(v1+v0))
        smd = (m1-m0)/s if s>0 else 0.0
        rows.append({"covariate": c, "mean_treat": m1, "mean_ctrl": m0, "smd": smd})
    return pd.DataFrame(rows).sort_values("covariate")

def srm_pvalue(n_treat: int, n_control: int, expected_ratio: float=1.0) -> float:
    n = n_treat + n_control
    if n == 0: return 1.0
    exp_treat = n * expected_ratio/(1+expected_ratio)
    exp_ctrl  = n - exp_treat
    obs = np.array([n_treat, n_control], dtype=float)
    exp = np.array([exp_treat, exp_ctrl], dtype=float)
    chi = ((obs-exp)**2/np.where(exp>0,exp,1)).sum()
    from scipy.stats import chi2
    return float(1-chi2.cdf(chi, df=1))

# ---------- Logistic primary effect ----------
@dataclass
class LogitEffect:
    odds_ratio: float
    coef: float
    se: float
    z: float
    pvalue: float
    ci_low: float
    ci_high: float

def logit_effect(df: pd.DataFrame, y: str="conversion", g: str="treatment", covars: Optional[Iterable[str]]=None,
                 cluster: Optional[str]=None) -> LogitEffect:
    X = pd.DataFrame({ "intercept": 1.0, g: df[g].astype(float) })
    if covars:
        for c in covars:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    model = sm.GLM(df[y].astype(float), X, family=sm.families.Binomial())
    if cluster:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    else:
        res = model.fit(cov_type="HC1")
    coef = float(res.params[g])
    se   = float(res.bse[g])
    z    = coef / se if se>0 else np.nan
    from scipy.stats import norm
    p    = float(2*(1-norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    lo, hi = coef - 1.96*se, coef + 1.96*se
    return LogitEffect(odds_ratio=float(np.exp(coef)), coef=coef, se=se, z=z, pvalue=p,
                       ci_low=float(np.exp(lo)), ci_high=float(np.exp(hi)))

def primary_effect_summary(df: pd.DataFrame, out_csv: Optional[str]=None) -> pd.DataFrame:
    r = two_prop_test(df)
    # CI for absolute difference (unpooled)
    n1 = int((df["treatment"]==1).sum())
    n0 = int((df["treatment"]==0).sum())
    var_diff = (r.p_treat*(1-r.p_treat))/max(n1,1) + (r.p_control*(1-r.p_control))/max(n0,1)
    se_diff = float(np.sqrt(max(var_diff, 0.0)))
    diff_lo, diff_hi = r.diff - 1.96*se_diff, r.diff + 1.96*se_diff

    le = logit_effect(df)
    out = pd.DataFrame([{
        "p_control": r.p_control,
        "p_treat": r.p_treat,
        "diff": r.diff,
        "diff_ci_low": diff_lo,
        "diff_ci_high": diff_hi,
        "z": r.z,
        "pvalue": r.pvalue,
        "odds_ratio": le.odds_ratio,
        "or_ci_low": le.ci_low,
        "or_ci_high": le.ci_high
    }])
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        logger.info("Wrote primary effect summary -> %s", out_csv)
    return out

if __name__ == "__main__":
    import argparse, sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add .../src
    from ab_uplift.data_loader import make_demo_df, load_and_validate
    p = argparse.ArgumentParser(description="A/B basics: two-proportion test, SMDs, and logit effect")
    p.add_argument("--data", type=str, default=None, help="Dataset path")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    p.add_argument("--out", type=str, default="reports/tables/primary_effect_summary.csv")
    args = p.parse_args()

    df = make_demo_df(20000) if args.demo else load_and_validate(args.data)
    print(primary_effect_summary(df, out_csv=args.out))
    covars = [f"f{i}" for i in range(12) if f"f{i}" in df.columns]
    smds = smd_table(df, covars)
    print(smds.head())
