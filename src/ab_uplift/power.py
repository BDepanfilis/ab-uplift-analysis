
#!/usr/bin/env python
"""
Power analysis for two-proportion tests (Cohen's h, required N, achieved power).
"""
from __future__ import annotations

from math import ceil
from dataclasses import dataclass
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

@dataclass
class PowerResult:
    required_n_control: int
    required_n_treat: int
    effect_size_h: float
    achieved_power: float

def required_n_two_prop(p_baseline: float, mde: float, *, mde_is_relative: bool=True,
                        alpha: float=0.05, power: float=0.8, ratio: float=1.0) -> tuple[int,int,float]:
    if not (0 < p_baseline < 1):
        raise ValueError("p_baseline must be in (0,1)")
    if mde <= 0: raise ValueError("mde must be > 0")
    if ratio <= 0: raise ValueError("ratio must be > 0")
    p1 = float(np.clip(p_baseline, 1e-12, 1-1e-12))
    p2 = p1*(1+mde) if mde_is_relative else p1 + mde
    p2 = float(np.clip(p2, 1e-12, 1-1e-12))
    es = proportion_effectsize(p2, p1)  # Cohen's h
    smp = NormalIndPower()
    n_c = smp.solve_power(effect_size=es, alpha=alpha, power=power, ratio=ratio, alternative="two-sided")
    n_t = n_c * ratio
    return ceil(n_c), ceil(n_t), float(es)

def achieved_power_two_prop(p_baseline: float, mde: float, n_control: int, n_treat: int, *,
                            mde_is_relative: bool=True, alpha: float=0.05) -> float:
    p1 = float(np.clip(p_baseline, 1e-12, 1-1e-12))
    p2 = p1*(1+mde) if mde_is_relative else p1 + mde
    p2 = float(np.clip(p2, 1e-12, 1-1e-12))
    es = proportion_effectsize(p2, p1)
    ratio = n_treat/max(n_control,1)
    smp = NormalIndPower()
    return float(smp.solve_power(effect_size=es, alpha=alpha, nobs1=n_control, ratio=ratio, alternative="two-sided"))

def analyze_power_from_df(df, *, alpha: float=0.05, power: float=0.8, mde_rel: float=0.1) -> PowerResult:
    import numpy as np
    y0 = df.loc[df["treatment"]==0, "conversion"].to_numpy()
    baseline = float(np.mean(y0)) if y0.size else 0.0
    n0 = int(np.sum(df["treatment"]==0))
    n1 = int(np.sum(df["treatment"]==1))
    ratio = n1/max(n0,1)
    n_req_c, n_req_t, es = required_n_two_prop(baseline, mde_rel, alpha=alpha, power=power, ratio=ratio)
    achieved = achieved_power_two_prop(baseline, mde_rel, n0, n1, alpha=alpha)
    return PowerResult(n_req_c, n_req_t, es, achieved)

if __name__ == "__main__":
    import argparse, pandas as pd, sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from src.data_loader import load_and_validate, make_demo_df
    p = argparse.ArgumentParser(description="Two-proportion power analysis")
    p.add_argument("--data", type=str, default=None, help="Dataset path")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--mde-rel", type=float, default=0.1)
    args = p.parse_args()
    df = make_demo_df(20000) if args.demo else load_and_validate(args.data)
    res = analyze_power_from_df(df, alpha=args.alpha, power=args.power, mde_rel=args.mde_rel)
    print(f"[power] required control={res.required_n_control:,} treat={res.required_n_treat:,} "
          f"h={res.effect_size_h:.4f} achieved_power={res.achieved_power:.3f}")
