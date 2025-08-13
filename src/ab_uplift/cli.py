#!/usr/bin/env python
"""
Console entrypoint for the ab_uplift pipeline.
Runs the same stages as scripts/run_all.py and logs each artifact.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .data_loader import load_and_validate, make_demo_df
from .power import analyze_power_from_df
from .ab_stats import primary_effect_summary, smd_table
from .cupac import run_cupac_gbm
from .causal import ipw_aipw_summary_cf
from .uplift import run_uplift_qini

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger("ab_uplift.cli")

def ensure_dirs():
    tables = Path("reports/tables"); tables.mkdir(parents=True, exist_ok=True)
    figs = Path("reports/figures"); figs.mkdir(parents=True, exist_ok=True)
    return tables, figs

def main():
    p = argparse.ArgumentParser(description="ab-uplift pipeline (CLI)")
    p.add_argument("--data", type=str, default=None, help="Path to dataset (parquet/csv).")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    p.add_argument("--rows", type=int, default=None, help="Cap rows for large files (e.g., 300000).")
    p.add_argument("--bins", type=int, default=20, help="Number of bins for Qini/gains.")
    args = p.parse_args()

    outdir, figdir = ensure_dirs()
    df = make_demo_df(60000) if args.demo else load_and_validate(args.data, n_rows=args.rows)

    # ---- Power ----
    pow_res = analyze_power_from_df(df)
    (outdir / "power_summary.csv").write_text(
        "required_n_control,required_n_treat,cohens_h,achieved_power\n"
        f"{pow_res.required_n_control},{pow_res.required_n_treat},"
        f"{pow_res.effect_size_h:.6f},{pow_res.achieved_power:.6f}\n"
    )
    logger.info("Wrote %s", outdir / "power_summary.csv")

    # ---- Primary effect + balance ----
    primary_effect_summary(df, out_csv=str(outdir / "primary_effect_summary.csv"))
    logger.info("Wrote %s", outdir / "primary_effect_summary.csv")
    covars = [c for c in df.columns if c.startswith("f")]
    smd_table(df, covars).to_csv(outdir / "smd_table.csv", index=False)
    logger.info("Wrote %s", outdir / "smd_table.csv")

    # ---- CUPAC ----
    run_cupac_gbm(df).to_csv(outdir / "cupac_summary_gbm.csv", index=False)
    logger.info("Wrote %s", outdir / "cupac_summary_gbm.csv")

    # ---- IPW / AIPW + propensity overlap plot ----
    ipw_summary, detail = ipw_aipw_summary_cf(df, return_detail=True)
    ipw_summary.to_csv(outdir / "ipw_aipw_summary.csv", index=False)
    logger.info("Wrote %s", outdir / "ipw_aipw_summary.csv")

    ps = detail["ps"]
    plt.figure()
    plt.hist(ps, bins=50)
    plt.axvline(0.05, linestyle="--")
    plt.axvline(0.95, linestyle="--")
    plt.xlabel("Propensity score")
    plt.ylabel("Frequency")
    plt.title("Propensity overlap")
    prop_png = figdir / "propensity_hist.png"
    plt.savefig(prop_png, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", prop_png)

    # ---- Uplift & Qini (+ AUUC) ----
    gains, qini_total, auuc = run_uplift_qini(df, bins=args.bins)
    gains.to_csv(outdir / "uplift_qini_gains.csv", index=False)
    (outdir / "uplift_qini_meta.csv").write_text(f"qini_total,{qini_total:.6f}\nauuc,{auuc:.6f}\n")
    logger.info("Wrote %s and %s", outdir / "uplift_qini_gains.csv", outdir / "uplift_qini_meta.csv")

    # Qini curve with random baseline
    n_users = int(df.shape[0])
    x_users = (gains["top_prop"] * n_users).to_numpy()
    y_qini = gains["qini"].to_numpy()
    plt.figure()
    plt.plot(x_users, y_qini, label="Qini curve")
    plt.plot([0, n_users], [0, y_qini[-1]], linestyle="--", label="Random baseline")
    plt.xlabel("Users (sorted by uplift)")
    plt.ylabel("Cumulative uplift")
    plt.title("Qini curve")
    plt.legend()
    qini_png = figdir / "qini_curve.png"
    plt.savefig(qini_png, dpi=160, bbox_inches="tight")
    plt.close()

    # Uplift gains by bin (bar of incremental Qini)
    inc = gains["qini"].diff().fillna(gains["qini"].iloc[0]).to_numpy()
    plt.figure()
    plt.bar(np.arange(1, len(inc)+1), inc)
    plt.xlabel("Bin (most likely → least)")
    plt.ylabel("Incremental conversions")
    plt.title(f"Uplift gains by bin (n_bins={args.bins})")
    gains_png = figdir / "uplift_gains_by_bin.png"
    plt.savefig(gains_png, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s and %s", qini_png, gains_png)

    # Top-K policy table
    top_props = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    qini_interp = np.interp(top_props, gains["top_prop"].to_numpy(), gains["qini"].to_numpy())
    policy = np.column_stack([top_props, qini_interp])
    np.savetxt(outdir / "uplift_topk_policy.csv", policy, delimiter=",",
               header="top_prop,qini", comments="", fmt="%.6f")
    logger.info("Wrote %s", outdir / "uplift_topk_policy.csv")

if __name__ == "__main__":
    main()
