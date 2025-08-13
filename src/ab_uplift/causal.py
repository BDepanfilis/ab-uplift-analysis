#!/usr/bin/env python
"""
IPW/AIPW with cross-fitting, analytic CIs, and overlap diagnostics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

@dataclass
class IPSummary:
    ate_ipw: float
    ate_aipw: float
    ess: float
    ess_frac: float
    clip_min: float
    clip_max: float
    ps_mean: float
    ps_min: float
    ps_max: float
    ps_pct_out_005_095: float
    ate_ipw_ci_low: float
    ate_ipw_ci_high: float
    ate_aipw_ci_low: float
    ate_aipw_ci_high: float
    n_rows: int
    n_rows_outcome_fit: int

def _propensity_cf(X: np.ndarray, t: np.ndarray, *, K: int=5, seed: int=7) -> np.ndarray:
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    ps = np.zeros_like(t, dtype=float)
    for tr, te in kf.split(X):
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=None)
        lr.fit(X[tr], t[tr])
        ps[te] = lr.predict_proba(X[te])[:,1]
    eps = 1e-6
    return np.clip(ps, eps, 1-eps)

def _ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float((w.sum()**2) / (np.sum(w**2) + 1e-12))

def ipw_aipw_summary_cf(
    df: pd.DataFrame,
    y: str="conversion",
    g: str="treatment",
    features: Optional[list[str]]=None,
    clip: tuple[float,float]=(0.01, 0.99),
    max_rows_for_outcome: int = 300_000,
    seed: int = 7,
    return_detail: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Cross-fitted IPW and AIPW with sklearn logistic outcome model.
    Returns (summary_df, detail) if return_detail=True else (summary_df, None).
    Detail dict includes 'ps' and 'ps_clip' arrays for diagnostics.
    """
    feats = features if features is not None else [c for c in df.columns if c.startswith("f")]
    X_full = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy()
    t_full = df[g].to_numpy().astype(int)
    y_full = df[y].to_numpy().astype(float)

    # Propensity on all rows
    ps = _propensity_cf(X_full, t_full, K=5, seed=seed)
    ps_clip = np.clip(ps, clip[0], clip[1])

    # Outcome model (logistic with t as feature), possibly subsample to fit
    n = len(df)
    if n > max_rows_for_outcome:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_rows_for_outcome, replace=False)
        X_fit = X_full[idx]
        t_fit = t_full[idx]
        y_fit = y_full[idx]
    else:
        X_fit, t_fit, y_fit = X_full, t_full, y_full

    X_with_t = np.column_stack([X_fit, t_fit]).astype("float32")
    lr_y = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=None)
    lr_y.fit(X_with_t, y_fit.astype(int))

    # Predict m1, m0 for all rows using learned coefficients
    coef = lr_y.coef_.ravel().astype("float64")
    intercept = float(lr_y.intercept_[0])
    beta_x = coef[:-1]
    beta_t = coef[-1]
    lin_x = intercept + X_full @ beta_x
    m1 = 1.0 / (1.0 + np.exp(-(lin_x + beta_t)))
    m0 = 1.0 / (1.0 + np.exp(-(lin_x)))

    # IPW
    w_t = t_full/ps_clip
    w_c = (1-t_full)/(1-ps_clip)
    z_ipw = y_full * w_t - y_full * w_c
    ate_ipw = float(np.mean(z_ipw))

    # AIPW
    term_t = (t_full * (y_full - m1))/ps_clip + m1
    term_c = ((1-t_full) * (y_full - m0))/(1-ps_clip) + m0
    psi = term_t - term_c
    ate_aipw = float(np.mean(psi))

    # Analytic CIs via influence function variance
    n_float = float(n)
    def ci_from_terms(terms: np.ndarray) -> tuple[float,float]:
        mu = float(np.mean(terms))
        se = float(np.std(terms, ddof=1) / np.sqrt(n_float))
        return mu - 1.96*se, mu + 1.96*se

    ate_ipw_ci_low, ate_ipw_ci_high = ci_from_terms(z_ipw)
    ate_aipw_ci_low, ate_aipw_ci_high = ci_from_terms(psi)

    ess = _ess(w_t + w_c)
    ess_frac = float(ess / max(n,1))
    pct_out = float(100.0 * np.mean((ps < 0.05) | (ps > 0.95)))

    summ = pd.DataFrame([{
        "ate_ipw": ate_ipw,
        "ate_aipw": ate_aipw,
        "ate_ipw_ci_low": ate_ipw_ci_low,
        "ate_ipw_ci_high": ate_ipw_ci_high,
        "ate_aipw_ci_low": ate_aipw_ci_low,
        "ate_aipw_ci_high": ate_aipw_ci_high,
        "ess": ess,
        "ess_frac": ess_frac,
        "clip_min": clip[0],
        "clip_max": clip[1],
        "ps_mean": float(np.mean(ps)),
        "ps_min": float(np.min(ps)),
        "ps_max": float(np.max(ps)),
        "ps_pct_out_005_095": pct_out,
        "n_rows": int(n),
        "n_rows_outcome_fit": int(len(X_fit)),
    }])

    detail = {"ps": ps, "ps_clip": ps_clip} if return_detail else None
    return summ, detail

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add .../src
    from ab_uplift.data_loader import make_demo_df, load_and_validate
    p = argparse.ArgumentParser(description="IPW/AIPW (cross-fitted) with diagnostics")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", type=str, default="reports/tables/ipw_aipw_summary.csv")
    p.add_argument("--rows", type=int, default=None, help="optional row cap at load")
    p.add_argument("--outcome-cap", type=int, default=300_000, help="max rows to fit outcome model")
    args = p.parse_args()
    df = make_demo_df(12000) if args.demo else load_and_validate(args.data, n_rows=args.rows)
    res, _ = ipw_aipw_summary_cf(df, max_rows_for_outcome=args.outcome_cap, return_detail=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res)
