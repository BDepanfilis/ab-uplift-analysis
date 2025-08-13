#!/usr/bin/env python
"""
T-learner uplift (OOF) with Qini curve, AUUC/Qini coefficient, and configurable bins.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def t_learner_uplift_oof(df: pd.DataFrame, y: str="conversion", g: str="treatment",
                         feature_cols: Optional[list[str]]=None, *, folds: int=5, seed: int=7) -> np.ndarray:
    feats = feature_cols if feature_cols is not None else [c for c in df.columns if c.startswith("f")]
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy()
    t = df[g].to_numpy().astype(int)
    yv = df[y].to_numpy().astype(int)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    uplift = np.zeros(len(df), dtype=float)
    for tr, te in kf.split(X):
        def fit_and_pred(subset_value: int) -> np.ndarray:
            idx = np.where(t[tr]==subset_value)[0]
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(X[tr][idx], yv[tr][idx])
            return lr.predict_proba(X[te])[:,1]
        p1 = fit_and_pred(1)
        p0 = fit_and_pred(0)
        uplift[te] = p1 - p0
    return uplift

def qini_and_gains(uplift: np.ndarray, y: np.ndarray, t: np.ndarray, *, bins: int=10) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute cumulative uplift by bins of predicted uplift.
    Returns (gains_df, qini_at_100pct, AUUC).
    AUUC is area under Qini vs top_prop in [0,1] using trapezoidal rule.
    """
    idx = np.argsort(-uplift)
    y = y[idx]; t = t[idx]; u = uplift[idx]
    n = len(u)
    splits = np.linspace(0, n, bins+1).astype(int)
    rows = []
    for b in range(1, bins+1):
        hi = splits[b]
        y_b = y[:hi]; t_b = t[:hi]
        n_t = t_b.sum(); n_c = len(t_b) - n_t
        conv_t = y_b[t_b==1].sum(); conv_c = y_b[t_b==0].sum()
        exp_t = (n_t / max(n_t + n_c, 1)) * (conv_t + conv_c)
        qini = float(conv_t - exp_t)
        rows.append({"top_prop": hi/n, "qini": qini, "n_obs": hi})
    gains = pd.DataFrame(rows).sort_values("top_prop").reset_index(drop=True)
    # AUUC (normalized by prop domain width 1.0)
    auuc = float(np.trapz(gains["qini"].to_numpy(), gains["top_prop"].to_numpy()))
    qini_total = float(gains["qini"].iloc[-1])
    return gains, qini_total, auuc

def run_uplift_qini(df: pd.DataFrame, y: str="conversion", g: str="treatment",
                    features: Optional[list[str]]=None, *, bins: int=10) -> Tuple[pd.DataFrame, float, float]:
    upl = t_learner_uplift_oof(df, y=y, g=g, feature_cols=features)
    gains, qini_total, auuc = qini_and_gains(upl, df[y].to_numpy(), df[g].to_numpy(), bins=bins)
    return gains, qini_total, auuc

if __name__ == "__main__":
    import argparse, sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add .../src
    from ab_uplift.data_loader import make_demo_df, load_and_validate
    p = argparse.ArgumentParser(description="Uplift modeling (T-learner OOF) and Qini")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", type=str, default="reports/tables/uplift_qini_gains.csv")
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()
    df = make_demo_df(50000) if args.demo else load_and_validate(args.data)
    gains, qini_total, auuc = run_uplift_qini(df, bins=args.bins)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gains.to_csv(args.out, index=False)
    print(gains.head())
    print(f"[uplift] cumulative Qini at 100%: {qini_total:.3f}, AUUC={auuc:.3f}")
