# -*- coding: utf-8 -*-
"""
AIM-DSPP REAL BIOLOGICAL DATA STUDY
==================================

Two empirical applications for the manuscript:

1) MASS::epil
   Longitudinal two-week seizure counts from 59 patients with epilepsy.
   The pre-treatment 8-week seizure count defines an external baseline
   two-week intensity R_i = base_i / 4.  Subjects are split, so no patient
   appears in more than one of train/validation/test.

2) glmmTMB::Owls
   Repeated sibling-negotiation (begging-call) counts from barn-owl nests.
   A history-only Gamma-Poisson-shrunk nest rate is used as a directing
   baseline frailty; brood size is the exposure.  History outcomes are never
   reused for model fitting or test scoring.

Models fixed a priori for both applications:
  * Directing baseline only
  * Linear Poisson AIM
  * Spline Poisson AIM
  * Poisson gradient boosting
  * Neural AIM-DSPP

The script:
  * loads MASS::epil through statsmodels.datasets.get_rdataset with a CSV fallback;
  * loads glmmTMB::Owls directly from the official glmmTMB GitHub data file
    (Rdatasets does not currently mirror this dataset);
  * prints every substantive table inline;
  * displays every figure inline;
  * saves figures as PDF + PNG;
  * saves tables as CSV + LaTeX;
  * saves fitted model objects / neural state dictionaries;
  * writes machine-readable summaries and an auto-generated LaTeX results note;
  * creates AIM_DSPP_RealBiology_Outputs.zip and triggers download in Colab.

IMPORTANT SCIENTIFIC DESIGN NOTE
--------------------------------
The empirical applications use interval counts rather than exact event times.
Consequently, the code evaluates conditional Poisson calibration with randomized
PIT residuals, not continuous-time time-rescaling.  This distinction is
intentional and should be preserved in the paper.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import shutil
import warnings
import subprocess
import importlib
import urllib.request
import urllib.error
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

# -----------------------------------------------------------------------------
# Light dependency check (Colab normally already contains these packages)
# -----------------------------------------------------------------------------
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "joblib": "joblib",
    "pyreadr": "pyreadr",
}

for module_name, pip_name in REQUIRED.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"Installing missing package: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import poisson, kstest, probplot, spearmanr
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import torch
from torch import nn
import pyreadr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Reproducibility / configuration
# -----------------------------------------------------------------------------
SEED = 20260824
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT = Path("/content/AIM_DSPP_RealBiology_Outputs") if Path("/content").exists() else Path.cwd() / "AIM_DSPP_RealBiology_Outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
DAT = OUT / "data"
MOD = OUT / "models"
META = OUT / "metadata"
for p in [OUT, FIG, TAB, DAT, MOD, META]:
    p.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "seed": SEED,
    "device": str(DEVICE),
    "bootstrap_repetitions": 750,
    "pit_seed": SEED + 19,
    "permutation_repetitions": 100,
    "neural_restarts": 3,
    "neural_max_epochs": 1800,
    "neural_patience": 160,
    "epilepsy_subject_test_fraction": 0.20,
    "epilepsy_subject_validation_fraction_of_remaining": 0.20,
    "owls_history_fraction_within_nest": 0.35,
    "owls_validation_fraction_within_nest": 0.15,
    "owls_test_fraction_within_nest": 0.20,
    "owls_gamma_poisson_prior_exposure": 10.0,
    "benchmark_set_fixed_before_test_evaluation": True,
}

with open(META / "config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

print("=" * 96)
print("AIM-DSPP REAL BIOLOGICAL DATA STUDY")
print("=" * 96)
print("Device:", DEVICE)
print("Output directory:", OUT)
print("Benchmark set is fixed a priori; test performance does not determine model inclusion.")
print("Exact event times are unavailable; randomized PIT is used instead of time-rescaling.")

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def display_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "-" * 96)
    print(title)
    print("-" * 96)
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))


def sanitize_latex_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("%", "pct")


def save_table(df: pd.DataFrame, stem: str, title: str, index: bool = False) -> None:
    display_table(df, title)
    df.to_csv(TAB / f"{stem}.csv", index=index)
    try:
        latex = df.to_latex(index=index, escape=True, float_format=lambda x: f"{x:.6g}")
        with open(TAB / f"{stem}.tex", "w", encoding="utf-8") as f:
            f.write(latex)
    except Exception as e:
        print(f"LaTeX table export warning for {stem}: {e}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def poisson_nll(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    return float(np.mean(mu - y * np.log(mu) + gammaln(y + 1.0)))


def poisson_deviance_mean(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    term = np.where(y > 0, y * np.log(np.clip(y, 1e-12, None) / mu) - (y - mu), mu)
    return float(2.0 * np.mean(term))


def pearson_dispersion(y: np.ndarray, mu: np.ndarray, dof_correction: int = 1) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    denom = max(len(y) - dof_correction, 1)
    return float(np.sum((y - mu) ** 2 / mu) / denom)


def randomized_pit(y: np.ndarray, mu: np.ndarray, seed: int) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=int)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
    rng = np.random.default_rng(seed)
    v = rng.uniform(size=len(y))
    lower = poisson.cdf(y - 1, mu)
    upper = poisson.cdf(y, mu)
    u = lower + v * (upper - lower)
    stat, pval = kstest(u, "uniform")
    return u, float(stat), float(pval)


def metric_row(y: np.ndarray, mu: np.ndarray, model: str, pit_seed: int) -> Dict[str, float]:
    u, ks, p = randomized_pit(y, mu, pit_seed)
    return {
        "Model": model,
        "NLL": poisson_nll(y, mu),
        "PoissonDeviance": poisson_deviance_mean(y, mu),
        "RMSE": float(np.sqrt(mean_squared_error(y, mu))),
        "MAE": float(mean_absolute_error(y, mu)),
        "MeanObserved": float(np.mean(y)),
        "MeanPredicted": float(np.mean(mu)),
        "PredObsRatio": float(np.sum(mu) / max(np.sum(y), 1e-12)),
        "PearsonDispersion": pearson_dispersion(y, mu),
        "PIT_KS": ks,
        "PIT_p": p,
        "PIT_mean": float(np.mean(u)),
        "PIT_var": float(np.var(u, ddof=1)),
    }


def calibration_table(y: np.ndarray, pred_dict: Dict[str, np.ndarray], n_bins: int = 8) -> pd.DataFrame:
    frames = []
    y = np.asarray(y, dtype=float)
    for model, mu in pred_dict.items():
        tmp = pd.DataFrame({"y": y, "mu": np.asarray(mu, dtype=float)})
        try:
            tmp["bin"] = pd.qcut(tmp["mu"], q=n_bins, duplicates="drop")
        except Exception:
            tmp["bin"] = pd.cut(tmp["mu"], bins=n_bins, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(
            n=("y", "size"),
            mean_observed=("y", "mean"),
            mean_predicted=("mu", "mean"),
            sum_observed=("y", "sum"),
            sum_predicted=("mu", "sum"),
        ).reset_index(drop=True)
        g.insert(0, "CalibrationBin", np.arange(1, len(g) + 1))
        g.insert(0, "Model", model)
        g["ObservedPredictedRatio"] = g["sum_observed"] / np.clip(g["sum_predicted"], 1e-12, None)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def cluster_bootstrap_metrics(
    df_test: pd.DataFrame,
    y_col: str,
    cluster_col: str,
    pred_cols: Dict[str, str],
    B: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clusters = np.array(sorted(df_test[cluster_col].astype(str).unique()))
    cluster_to_idx = {
        c: np.where(df_test[cluster_col].astype(str).values == c)[0]
        for c in clusters
    }
    boot = {m: {"NLL": [], "PoissonDeviance": [], "RMSE": [], "MAE": []} for m in pred_cols}
    paired = {m: [] for m in pred_cols if m != "Neural AIM-DSPP"}

    y_all = df_test[y_col].to_numpy(float)
    pred_arr = {m: df_test[col].to_numpy(float) for m, col in pred_cols.items()}

    for _ in range(B):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        y = y_all[idx]
        for m, muall in pred_arr.items():
            mu = muall[idx]
            boot[m]["NLL"].append(poisson_nll(y, mu))
            boot[m]["PoissonDeviance"].append(poisson_deviance_mean(y, mu))
            boot[m]["RMSE"].append(float(np.sqrt(mean_squared_error(y, mu))))
            boot[m]["MAE"].append(float(mean_absolute_error(y, mu)))
        if "Neural AIM-DSPP" in pred_arr:
            neural_nll = poisson_nll(y, pred_arr["Neural AIM-DSPP"][idx])
            for m in paired:
                paired[m].append(poisson_nll(y, pred_arr[m][idx]) - neural_nll)

    rows = []
    for m in pred_cols:
        for metric, vals in boot[m].items():
            vals = np.asarray(vals)
            rows.append({
                "Model": m,
                "Metric": metric,
                "BootstrapMean": vals.mean(),
                "CI2.5": np.quantile(vals, 0.025),
                "CI97.5": np.quantile(vals, 0.975),
            })
    ci_df = pd.DataFrame(rows)

    pair_rows = []
    for m, vals in paired.items():
        vals = np.asarray(vals)
        pair_rows.append({
            "Benchmark": m,
            "Comparison": "Benchmark NLL - Neural NLL",
            "MeanDifference": vals.mean(),
            "CI2.5": np.quantile(vals, 0.025),
            "CI97.5": np.quantile(vals, 0.975),
            "BootstrapFractionNeuralBetter": np.mean(vals > 0),
        })
    paired_df = pd.DataFrame(pair_rows)
    return ci_df, paired_df


def group_aggregate_table(
    df_test: pd.DataFrame,
    group_col: str,
    y_col: str,
    pred_cols: Dict[str, str],
) -> pd.DataFrame:
    agg = df_test.groupby(group_col, observed=True)[y_col].sum().rename("Observed").to_frame()
    for model, col in pred_cols.items():
        agg[model] = df_test.groupby(group_col, observed=True)[col].sum()
    agg = agg.reset_index()
    return agg


def save_versions() -> None:
    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsmodels": sm.__version__,
        "torch": torch.__version__,
    }
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    with open(META / "software_versions.json", "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)

save_versions()

# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------
# MASS::epil is available through statsmodels/Rdatasets.  glmmTMB::Owls is
# NOT currently mirrored by Rdatasets, so attempting get_rdataset("Owls",
# "glmmTMB") produces a 404.  We therefore load the owl data from the
# official glmmTMB repository, where the package's Owls.rda file lives.

def load_rdataset(name: str, package: str, fallback_url: str) -> Tuple[pd.DataFrame, str, str]:
    """Load an R dataset via statsmodels, with a CSV fallback when available."""
    try:
        ds = sm.datasets.get_rdataset(name, package=package, cache=True)
        df = ds.data.copy()
        doc = str(ds.__doc__)
        route = f"statsmodels.datasets.get_rdataset('{name}', package='{package}')"
        return df, route, doc
    except Exception as e:
        print(f"statsmodels get_rdataset failed for {package}::{name}: {e}")
        print("Trying verified CSV fallback ...")
        df = pd.read_csv(fallback_url)
        for candidate in ["rownames", "Row.names", "Unnamed: 0"]:
            if candidate in df.columns:
                df = df.drop(columns=[candidate])
        route = fallback_url
        doc = f"Fallback URL used: {fallback_url}"
        return df, route, doc


def _read_owls_rda(path: Path) -> pd.DataFrame:
    """Read Owls.rda and robustly identify the data frame inside it."""
    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"No R objects were found in {path}.")

    expected = {
        "Nest", "FoodTreatment", "SexParent", "ArrivalTime",
        "SiblingNegotiation", "BroodSize"
    }

    # Normal case: the object stored in the RDA is named 'Owls'.
    if "Owls" in result:
        df = result["Owls"].copy()
        if expected.issubset(df.columns):
            return df

    # Defensive fallback: choose whichever stored data frame has the
    # documented glmmTMB::Owls columns.
    for key, obj in result.items():
        if isinstance(obj, pd.DataFrame) and expected.issubset(obj.columns):
            print(f"Owls object was stored under R object name {key!r}; using it.")
            return obj.copy()

    raise ValueError(
        "The downloaded RDA was readable, but it did not contain the documented "
        "glmmTMB::Owls variables. Objects found: " + ", ".join(map(str, result.keys()))
    )


def load_owls_official() -> Tuple[pd.DataFrame, str, str]:
    """
    Load glmmTMB::Owls from the official glmmTMB GitHub repository.

    Several routes are attempted because Colab/network environments occasionally
    handle GitHub raw-file redirects differently.  Every route points to the
    same official glmmTMB repository; there is no synthetic-data fallback.
    """
    cache_dir = Path("/content") if Path("/content").exists() else OUT
    local_path = cache_dir / "Owls_official_glmmTMB.rda"

    expected_n = 599
    documentation = (
        "glmmTMB::Owls — Begging by Owl Nestlings; 599 observations. "
        "Official package documentation: https://glmmtmb.github.io/glmmTMB/reference/Owls.html"
    )

    # Reuse a previously downloaded valid file if a cell/script is rerun.
    if local_path.exists() and local_path.stat().st_size > 0:
        try:
            df = _read_owls_rda(local_path)
            if len(df) == expected_n:
                return df, f"local cache of official glmmTMB Owls.rda: {local_path}", documentation
        except Exception:
            try:
                local_path.unlink()
            except Exception:
                pass

    raw_urls = [
        "https://raw.githubusercontent.com/glmmTMB/glmmTMB/master/glmmTMB/data/Owls.rda",
        "https://github.com/glmmTMB/glmmTMB/raw/refs/heads/master/glmmTMB/data/Owls.rda",
    ]
    errors = []

    for url in raw_urls:
        try:
            print(f"Downloading official glmmTMB::Owls from:\n  {url}")
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 AIM-DSPP-research-script",
                    "Accept": "application/octet-stream,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise ValueError(f"Downloaded payload is unexpectedly small ({len(payload)} bytes).")
            local_path.write_bytes(payload)
            df = _read_owls_rda(local_path)
            if len(df) != expected_n:
                raise ValueError(f"Expected {expected_n} Owls rows but obtained {len(df)}.")
            print(f"Owls loaded successfully: {df.shape[0]} rows x {df.shape[1]} columns.")
            return df, url, documentation
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
            print(f"  route failed: {type(e).__name__}: {e}")

    # Final official-repository fallback: GitHub Contents API returns the same
    # package data file base64-encoded.
    api_url = (
        "https://api.github.com/repos/glmmTMB/glmmTMB/contents/"
        "glmmTMB/data/Owls.rda?ref=master"
    )
    try:
        print("Trying official GitHub Contents API fallback ...")
        req = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": "Mozilla/5.0 AIM-DSPP-research-script",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            meta = json.loads(response.read().decode("utf-8"))
        if meta.get("encoding") != "base64" or "content" not in meta:
            raise ValueError("GitHub API response did not contain base64 file content.")
        payload = base64.b64decode(meta["content"])
        local_path.write_bytes(payload)
        df = _read_owls_rda(local_path)
        if len(df) != expected_n:
            raise ValueError(f"Expected {expected_n} Owls rows but obtained {len(df)}.")
        print(f"Owls loaded successfully: {df.shape[0]} rows x {df.shape[1]} columns.")
        return df, api_url, documentation
    except Exception as e:
        errors.append(f"{api_url}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Could not download the official glmmTMB::Owls data file. This is a network/download "
        "problem, not a model-fitting problem. Tried only official glmmTMB GitHub routes.\n"
        + "\n".join("  - " + x for x in errors)
    )


EPIL_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"

epil_raw, epil_route, epil_doc = load_rdataset("epil", "MASS", EPIL_URL)
owls_raw, owls_route, owls_doc = load_owls_official()

with open(META / "epil_dataset_documentation.txt", "w", encoding="utf-8") as f:
    f.write(epil_doc)
with open(META / "owls_dataset_documentation.txt", "w", encoding="utf-8") as f:
    f.write(owls_doc)

print("\nDataset loading routes:")
print("  Epilepsy:", epil_route)
print("  Owls:    ", owls_route)

# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

def prepare_epilepsy(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    required = {"y", "trt", "base", "age", "subject", "period"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"epil is missing expected columns: {sorted(missing)}")
    d["subject"] = d["subject"].astype(str)
    d["trt"] = d["trt"].astype(str)
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d["base"] = pd.to_numeric(d["base"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["period"] = pd.to_numeric(d["period"], errors="coerce").astype(int)
    d = d.dropna(subset=["y", "base", "age", "period"]).reset_index(drop=True)
    d["progabide"] = d["trt"].str.lower().str.contains("prog").astype(int)
    d["log_base"] = np.log1p(d["base"])
    d["baseline_two_week_rate"] = np.clip(d["base"] / 4.0, 1e-4, None)
    d["R"] = d["baseline_two_week_rate"]
    d["row_id"] = np.arange(len(d))
    return d


def stratified_subject_split(d: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subject_info = d.groupby("subject", observed=True).agg(progabide=("progabide", "first")).reset_index()
    train_subjects, val_subjects, test_subjects = [], [], []
    for trt in sorted(subject_info["progabide"].unique()):
        s = subject_info.loc[subject_info["progabide"] == trt, "subject"].to_numpy()
        rng.shuffle(s)
        n = len(s)
        n_test = max(1, int(round(CONFIG["epilepsy_subject_test_fraction"] * n)))
        rem = n - n_test
        n_val = max(1, int(round(CONFIG["epilepsy_subject_validation_fraction_of_remaining"] * rem)))
        test_subjects.extend(s[:n_test].tolist())
        val_subjects.extend(s[n_test:n_test+n_val].tolist())
        train_subjects.extend(s[n_test+n_val:].tolist())
    d = d.copy()
    d["split"] = "train"
    d.loc[d["subject"].isin(val_subjects), "split"] = "validation"
    d.loc[d["subject"].isin(test_subjects), "split"] = "test"
    return d


def prepare_owls(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    required = {"Nest", "FoodTreatment", "SexParent", "ArrivalTime", "SiblingNegotiation", "BroodSize"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Owls is missing expected columns: {sorted(missing)}")
    d["Nest"] = d["Nest"].astype(str)
    d["FoodTreatment"] = d["FoodTreatment"].astype(str)
    d["SexParent"] = d["SexParent"].astype(str)
    d["ArrivalTime"] = pd.to_numeric(d["ArrivalTime"], errors="coerce")
    d["SiblingNegotiation"] = pd.to_numeric(d["SiblingNegotiation"], errors="coerce")
    d["BroodSize"] = pd.to_numeric(d["BroodSize"], errors="coerce")
    d = d.dropna(subset=["ArrivalTime", "SiblingNegotiation", "BroodSize"]).reset_index(drop=True)
    d = d[d["BroodSize"] > 0].reset_index(drop=True)
    d["y"] = d["SiblingNegotiation"].astype(float)
    d["deprived"] = d["FoodTreatment"].str.lower().str.contains("depriv").astype(int)
    d["male_parent"] = d["SexParent"].str.lower().str.contains("male").astype(int)
    d["log_brood"] = np.log(d["BroodSize"])
    d["row_id"] = np.arange(len(d))

    # Within each nest: history is reserved only for frailty estimation.
    # Remaining rows are assigned to train/validation/test.
    rng = np.random.default_rng(seed)
    d["split"] = "train"
    for nest in sorted(d["Nest"].unique()):
        idx = d.index[d["Nest"] == nest].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 5:
            # Very small groups: one history, one validation, one test if possible.
            n_hist, n_val, n_test = 1, 1 if n >= 3 else 0, 1 if n >= 2 else 0
        else:
            n_hist = max(2, int(round(CONFIG["owls_history_fraction_within_nest"] * n)))
            n_test = max(1, int(round(CONFIG["owls_test_fraction_within_nest"] * n)))
            n_val = max(1, int(round(CONFIG["owls_validation_fraction_within_nest"] * n)))
            while n_hist + n_test + n_val >= n and n_hist > 1:
                n_hist -= 1
            while n_hist + n_test + n_val >= n and n_val > 1:
                n_val -= 1
        d.loc[idx[:n_hist], "split"] = "history"
        d.loc[idx[n_hist:n_hist+n_val], "split"] = "validation"
        d.loc[idx[n_hist+n_val:n_hist+n_val+n_test], "split"] = "test"
        d.loc[idx[n_hist+n_val+n_test:], "split"] = "train"

    hist = d[d["split"] == "history"].copy()
    global_rate = hist["y"].sum() / hist["BroodSize"].sum()
    kappa = CONFIG["owls_gamma_poisson_prior_exposure"]
    nest_hist = hist.groupby("Nest", observed=True).agg(
        hist_calls=("y", "sum"),
        hist_exposure=("BroodSize", "sum"),
        hist_n=("y", "size"),
    )
    nest_hist["raw_history_rate"] = nest_hist["hist_calls"] / nest_hist["hist_exposure"]
    nest_hist["shrunk_nest_rate"] = (
        nest_hist["hist_calls"] + kappa * global_rate
    ) / (
        nest_hist["hist_exposure"] + kappa
    )
    d = d.merge(nest_hist[["hist_calls", "hist_exposure", "hist_n", "raw_history_rate", "shrunk_nest_rate"]],
                left_on="Nest", right_index=True, how="left")
    d["R"] = np.clip(d["BroodSize"] * d["shrunk_nest_rate"], 1e-5, None)
    d.attrs["global_history_rate"] = float(global_rate)
    return d


epil = prepare_epilepsy(epil_raw)
epil = stratified_subject_split(epil, SEED + 1)
owls = prepare_owls(owls_raw, SEED + 2)

epil.to_csv(DAT / "epilepsy_prepared.csv", index=False)
owls.to_csv(DAT / "owls_prepared.csv", index=False)

# -----------------------------------------------------------------------------
# Descriptive tables
# -----------------------------------------------------------------------------

dataset_overview = pd.DataFrame([
    {
        "Dataset": "MASS::epil",
        "Biological response": "Two-week epileptic seizure count",
        "Rows": len(epil),
        "Biological units": epil["subject"].nunique(),
        "Unit": "patient",
        "Directing baseline R": "pre-treatment 8-week count / 4",
        "Exposure": "2 weeks per row",
        "Python route": epil_route,
        "Primary source DOI": "10.2307/2532086",
    },
    {
        "Dataset": "glmmTMB::Owls",
        "Biological response": "Sibling-negotiation / begging-call count",
        "Rows": len(owls),
        "Biological units": owls["Nest"].nunique(),
        "Unit": "nest",
        "Directing baseline R": "history-only shrunk nest rate x brood size",
        "Exposure": "brood size",
        "Python route": owls_route,
        "Primary source DOI": "10.1016/j.anbehav.2007.01.027",
    },
])
save_table(dataset_overview, "00_dataset_overview", "Table 0. Biological datasets and AIM-DSPP empirical interpretation")


epil_split = epil.groupby("split", observed=True).agg(
    rows=("y", "size"),
    subjects=("subject", "nunique"),
    mean_count=("y", "mean"),
    variance_count=("y", "var"),
    mean_baseline_R=("R", "mean"),
).reset_index()
epil_split["variance_mean_ratio"] = epil_split["variance_count"] / epil_split["mean_count"]
save_table(epil_split, "01_epilepsy_split_summary", "Table 1. Epilepsy split and count summary")


epil_disp = epil.groupby(["trt", "period"], observed=True).agg(
    n=("y", "size"), mean=("y", "mean"), variance=("y", "var"), zeros=("y", lambda x: np.mean(np.asarray(x) == 0)),
    mean_R=("R", "mean")
).reset_index()
epil_disp["variance_mean_ratio"] = epil_disp["variance"] / epil_disp["mean"]
save_table(epil_disp, "02_epilepsy_dispersion_by_treatment_period", "Table 2. Epilepsy dispersion by treatment and period")


owls_split = owls.groupby("split", observed=True).agg(
    rows=("y", "size"), nests=("Nest", "nunique"), mean_count=("y", "mean"), variance_count=("y", "var"),
    mean_R=("R", "mean"), mean_brood=("BroodSize", "mean")
).reset_index()
owls_split["variance_mean_ratio"] = owls_split["variance_count"] / owls_split["mean_count"]
save_table(owls_split, "03_owls_split_summary", "Table 3. Owl history/train/validation/test split summary")

owls_disp = owls.groupby(["FoodTreatment", "SexParent"], observed=True).agg(
    n=("y", "size"), mean=("y", "mean"), variance=("y", "var"), zeros=("y", lambda x: np.mean(np.asarray(x) == 0)),
    mean_brood=("BroodSize", "mean")
).reset_index()
owls_disp["variance_mean_ratio"] = owls_disp["variance"] / owls_disp["mean"]
save_table(owls_disp, "04_owls_dispersion_by_food_parent", "Table 4. Owl count dispersion by food treatment and parent sex")

nest_frailty = owls[["Nest", "hist_calls", "hist_exposure", "hist_n", "raw_history_rate", "shrunk_nest_rate"]].drop_duplicates("Nest")
nest_frailty = nest_frailty.sort_values("shrunk_nest_rate").reset_index(drop=True)
save_table(nest_frailty, "05_owls_history_frailty", "Table 5. History-only nest frailty estimates")

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

@dataclass
class DatasetDesign:
    name: str
    df: pd.DataFrame
    y_col: str
    cluster_col: str
    continuous_cols: List[str]
    binary_cols: List[str]
    linear_cols: List[str]
    neural_cols: List[str]
    train_label: str = "train"
    val_label: str = "validation"
    test_label: str = "test"


def add_epil_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    for p in [2, 3, 4]:
        x[f"period_{p}"] = (x["period"] == p).astype(int)
        x[f"progabide_period_{p}"] = x["progabide"] * x[f"period_{p}"]
    return x


def add_owl_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["food_parent_interaction"] = x["deprived"] * x["male_parent"]
    x["arrival_deprived"] = x["ArrivalTime"] * x["deprived"]
    x["arrival_male"] = x["ArrivalTime"] * x["male_parent"]
    return x


epil = add_epil_features(epil)
owls = add_owl_features(owls)

epil_design = DatasetDesign(
    name="Epilepsy",
    df=epil,
    y_col="y",
    cluster_col="subject",
    continuous_cols=["age", "log_base", "period"],
    binary_cols=["progabide"],
    linear_cols=["age", "log_base", "progabide", "period_2", "period_3", "period_4",
                 "progabide_period_2", "progabide_period_3", "progabide_period_4"],
    neural_cols=["age", "log_base", "period", "progabide"],
)

owl_design = DatasetDesign(
    name="Owls",
    df=owls,
    y_col="y",
    cluster_col="Nest",
    continuous_cols=["ArrivalTime", "log_brood"],
    binary_cols=["deprived", "male_parent", "food_parent_interaction"],
    linear_cols=["ArrivalTime", "log_brood", "deprived", "male_parent", "food_parent_interaction",
                 "arrival_deprived", "arrival_male"],
    neural_cols=["ArrivalTime", "log_brood", "deprived", "male_parent"],
)

# -----------------------------------------------------------------------------
# Model classes / fitting
# -----------------------------------------------------------------------------

class BaselineOnlyModel:
    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(df["R"].to_numpy(float), 1e-8, None)

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(df), dtype=float)


class SklearnRatioModel:
    def __init__(self, estimator, feature_cols: List[str], name: str):
        self.estimator = estimator
        self.feature_cols = feature_cols
        self.name = name

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        q = self.estimator.predict(df[self.feature_cols])
        return np.clip(np.asarray(q, dtype=float), 1e-8, 1e8)

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(df["R"].to_numpy(float) * self.predict_ratio(df), 1e-8, None)


class SplineRatioModel:
    def __init__(self, estimator, continuous_cols: List[str], binary_cols: List[str], name: str):
        self.estimator = estimator
        self.continuous_cols = continuous_cols
        self.binary_cols = binary_cols
        self.feature_cols = continuous_cols + binary_cols
        self.name = name

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        q = self.estimator.predict(df[self.feature_cols])
        return np.clip(np.asarray(q, dtype=float), 1e-8, 1e8)

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(df["R"].to_numpy(float) * self.predict_ratio(df), 1e-8, None)


class AIMNet(nn.Module):
    def __init__(self, p: int, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralAIMModel:
    def __init__(self, scaler: StandardScaler, net: AIMNet, feature_cols: List[str], center: float, history: pd.DataFrame, restart_seed: int):
        self.scaler = scaler
        self.net = net.cpu().eval()
        self.feature_cols = feature_cols
        self.center = float(center)
        self.history = history
        self.restart_seed = int(restart_seed)

    @torch.no_grad()
    def raw_eta(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.feature_cols].to_numpy(float)).astype(np.float32)
        z = torch.from_numpy(X)
        return self.net(z).cpu().numpy()

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        eta = np.clip(self.raw_eta(df), -12, 12)
        return np.exp(eta)

    def centered_modulator(self, df: pd.DataFrame) -> np.ndarray:
        eta = np.clip(self.raw_eta(df) - self.center, -12, 12)
        return np.exp(eta)

    @property
    def global_scale(self) -> float:
        return float(np.exp(self.center))

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(df["R"].to_numpy(float) * self.predict_ratio(df), 1e-8, None)


def fit_linear_aim(design: DatasetDesign) -> Tuple[SklearnRatioModel, pd.DataFrame]:
    d = design.df
    tr = d[d["split"] == design.train_label]
    va = d[d["split"] == design.val_label]
    Xtr = tr[design.linear_cols]
    y_ratio = tr[design.y_col].to_numpy(float) / np.clip(tr["R"].to_numpy(float), 1e-8, None)
    w = tr["R"].to_numpy(float)
    records = []
    best = None
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=alpha, solver="newton-cholesky", max_iter=300, tol=1e-6)),
        ])
        pipe.fit(Xtr, y_ratio, model__sample_weight=w)
        mu_val = va["R"].to_numpy(float) * np.clip(pipe.predict(va[design.linear_cols]), 1e-8, None)
        nll = poisson_nll(va[design.y_col].to_numpy(float), mu_val)
        records.append({"alpha": alpha, "ValidationNLL": nll})
        if best is None or nll < best[0]:
            best = (nll, pipe, alpha)
    model = SklearnRatioModel(best[1], design.linear_cols, "Linear Poisson AIM")
    return model, pd.DataFrame(records)


def fit_spline_aim(design: DatasetDesign) -> Tuple[SplineRatioModel, pd.DataFrame]:
    d = design.df
    tr = d[d["split"] == design.train_label]
    va = d[d["split"] == design.val_label]
    cols = design.continuous_cols + design.binary_cols
    y_ratio = tr[design.y_col].to_numpy(float) / np.clip(tr["R"].to_numpy(float), 1e-8, None)
    w = tr["R"].to_numpy(float)
    records = []
    best = None
    for knots in [3, 4, 5]:
        for alpha in [1e-4, 1e-3, 1e-2, 0.1]:
            pre = ColumnTransformer([
                ("splines", Pipeline([
                    ("spline", SplineTransformer(n_knots=knots, degree=3, include_bias=False)),
                    ("scale", StandardScaler()),
                ]), design.continuous_cols),
                ("binary", "passthrough", design.binary_cols),
            ], remainder="drop")
            pipe = Pipeline([
                ("pre", pre),
                ("model", PoissonRegressor(alpha=alpha, solver="newton-cholesky", max_iter=400, tol=1e-6)),
            ])
            pipe.fit(tr[cols], y_ratio, model__sample_weight=w)
            mu_val = va["R"].to_numpy(float) * np.clip(pipe.predict(va[cols]), 1e-8, None)
            nll = poisson_nll(va[design.y_col].to_numpy(float), mu_val)
            records.append({"knots": knots, "alpha": alpha, "ValidationNLL": nll})
            if best is None or nll < best[0]:
                best = (nll, pipe, knots, alpha)
    model = SplineRatioModel(best[1], design.continuous_cols, design.binary_cols, "Spline Poisson AIM")
    return model, pd.DataFrame(records)


def fit_histgb_aim(design: DatasetDesign) -> Tuple[SklearnRatioModel, pd.DataFrame]:
    d = design.df
    tr = d[d["split"] == design.train_label]
    va = d[d["split"] == design.val_label]
    cols = design.neural_cols
    y_ratio = tr[design.y_col].to_numpy(float) / np.clip(tr["R"].to_numpy(float), 1e-8, None)
    w = tr["R"].to_numpy(float)
    records = []
    best = None
    grid = [
        (2, 0.03, 0.0), (2, 0.06, 1.0),
        (3, 0.03, 0.5), (3, 0.06, 1.0),
        (4, 0.03, 1.0), (4, 0.05, 2.0),
    ]
    for depth, lr, l2 in grid:
        est = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=lr, max_depth=depth,
            max_iter=450, l2_regularization=l2,
            min_samples_leaf=max(5, int(0.04 * len(tr))),
            random_state=SEED,
        )
        est.fit(tr[cols], y_ratio, sample_weight=w)
        mu_val = va["R"].to_numpy(float) * np.clip(est.predict(va[cols]), 1e-8, None)
        nll = poisson_nll(va[design.y_col].to_numpy(float), mu_val)
        records.append({"max_depth": depth, "learning_rate": lr, "l2": l2, "ValidationNLL": nll})
        if best is None or nll < best[0]:
            best = (nll, est, depth, lr, l2)
    model = SklearnRatioModel(best[1], cols, "Poisson Gradient Boosting")
    return model, pd.DataFrame(records)


def fit_neural_aim(design: DatasetDesign) -> Tuple[NeuralAIMModel, pd.DataFrame, pd.DataFrame]:
    d = design.df
    tr = d[d["split"] == design.train_label].copy()
    va = d[d["split"] == design.val_label].copy()
    cols = design.neural_cols

    scaler = StandardScaler().fit(tr[cols].to_numpy(float))
    Xtr = scaler.transform(tr[cols].to_numpy(float)).astype(np.float32)
    Xva = scaler.transform(va[cols].to_numpy(float)).astype(np.float32)
    Rtr = tr["R"].to_numpy(np.float32)
    Rva = va["R"].to_numpy(np.float32)
    ytr = tr[design.y_col].to_numpy(np.float32)
    yva = va[design.y_col].to_numpy(np.float32)

    tx = torch.from_numpy(Xtr).to(DEVICE)
    vx = torch.from_numpy(Xva).to(DEVICE)
    tR = torch.from_numpy(Rtr).to(DEVICE)
    vR = torch.from_numpy(Rva).to(DEVICE)
    ty = torch.from_numpy(ytr).to(DEVICE)

    restart_summary = []
    best_overall = None

    for restart in range(CONFIG["neural_restarts"]):
        rseed = SEED + 1000 + 31 * restart + (0 if design.name == "Epilepsy" else 500)
        torch.manual_seed(rseed)
        net = AIMNet(len(cols), 32, 16).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=0.008, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.55, patience=60, min_lr=1e-5)
        best_state = None
        best_val = np.inf
        best_epoch = 0
        wait = 0
        history = []

        for epoch in range(1, CONFIG["neural_max_epochs"] + 1):
            net.train()
            opt.zero_grad()
            eta = torch.clamp(net(tx), -10.0, 10.0)
            mu = torch.clamp(tR * torch.exp(eta), min=1e-8)
            loss = torch.mean(mu - ty * torch.log(mu))
            # Small curvature penalty via weight decay already acts as regularization.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()

            net.eval()
            with torch.no_grad():
                veta = torch.clamp(net(vx), -10.0, 10.0)
                vmu = (vR * torch.exp(veta)).detach().cpu().numpy()
                val_nll = poisson_nll(yva, vmu)
            scheduler.step(val_nll)
            lr_now = opt.param_groups[0]["lr"]
            history.append({"epoch": epoch, "TrainObjective": float(loss.item()), "ValidationNLL": val_nll, "LearningRate": lr_now})

            if val_nll < best_val - 1e-7:
                best_val = val_nll
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= CONFIG["neural_patience"]:
                break

        restart_summary.append({
            "Restart": restart + 1,
            "Seed": rseed,
            "BestEpoch": best_epoch,
            "BestValidationNLL": best_val,
            "EpochsRun": len(history),
        })
        if best_overall is None or best_val < best_overall[0]:
            best_overall = (best_val, best_state, pd.DataFrame(history), rseed)

    best_net = AIMNet(len(cols), 32, 16)
    best_net.load_state_dict(best_overall[1])
    best_net.eval()
    with torch.no_grad():
        train_eta = best_net(torch.from_numpy(Xtr)).numpy()
    center = float(np.mean(train_eta))
    model = NeuralAIMModel(scaler, best_net, cols, center, best_overall[2], best_overall[3])
    return model, pd.DataFrame(restart_summary), best_overall[2]


def fit_all_models(design: DatasetDesign) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    print("\n" + "=" * 96)
    print(f"FITTING FIXED MODEL SET: {design.name}")
    print("=" * 96)
    models: Dict[str, Any] = {"Directing baseline only": BaselineOnlyModel()}
    tuning: Dict[str, pd.DataFrame] = {}

    print("  -> Linear Poisson AIM", flush=True)
    models["Linear Poisson AIM"], tuning["linear"] = fit_linear_aim(design)
    print("  <- Linear done; -> Spline Poisson AIM", flush=True)
    models["Spline Poisson AIM"], tuning["spline"] = fit_spline_aim(design)
    print("  <- Spline done; -> Poisson Gradient Boosting", flush=True)
    models["Poisson Gradient Boosting"], tuning["histgb"] = fit_histgb_aim(design)
    print("  <- HGB done; -> Neural AIM-DSPP", flush=True)
    models["Neural AIM-DSPP"], tuning["neural_restarts"], tuning["neural_history"] = fit_neural_aim(design)
    print("  <- Neural done", flush=True)
    return models, tuning

# -----------------------------------------------------------------------------
# Fit both applications
# -----------------------------------------------------------------------------

epil_models, epil_tuning = fit_all_models(epil_design)
owl_models, owl_tuning = fit_all_models(owl_design)

for name, t in epil_tuning.items():
    save_table(t, f"06_epilepsy_tuning_{name}", f"Epilepsy tuning diagnostic: {name}")
for name, t in owl_tuning.items():
    save_table(t, f"07_owls_tuning_{name}", f"Owl tuning diagnostic: {name}")

# Save model objects
for tag, models in [("epilepsy", epil_models), ("owls", owl_models)]:
    for model_name, model in models.items():
        stem = sanitize_latex_name(model_name.lower())
        if isinstance(model, NeuralAIMModel):
            torch.save(model.net.state_dict(), MOD / f"{tag}_{stem}_state_dict.pt")
            payload = {
                "feature_cols": model.feature_cols,
                "scaler_mean": model.scaler.mean_.tolist(),
                "scaler_scale": model.scaler.scale_.tolist(),
                "normalization_center_log_ratio": model.center,
                "global_scale_exp_center": model.global_scale,
                "restart_seed": model.restart_seed,
            }
            with open(MOD / f"{tag}_{stem}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        elif not isinstance(model, BaselineOnlyModel):
            joblib.dump(model.estimator, MOD / f"{tag}_{stem}.joblib")

# -----------------------------------------------------------------------------
# Evaluate on untouched test sets
# -----------------------------------------------------------------------------

def evaluate_design(design: DatasetDesign, models: Dict[str, Any], tag: str) -> Dict[str, Any]:
    d = design.df.copy()
    test = d[d["split"] == design.test_label].copy().reset_index(drop=True)
    pred_cols = {}
    for j, (name, model) in enumerate(models.items()):
        col = f"pred_{j+1:02d}"
        test[col] = model.predict_mu(test)
        test[f"ratio_{j+1:02d}"] = model.predict_ratio(test)
        pred_cols[name] = col

    test.to_csv(DAT / f"{tag}_test_predictions.csv", index=False)

    rows = []
    pits = {}
    for j, (name, col) in enumerate(pred_cols.items()):
        y = test[design.y_col].to_numpy(float)
        mu = test[col].to_numpy(float)
        rows.append(metric_row(y, mu, name, CONFIG["pit_seed"] + 100*j + (0 if tag == "epilepsy" else 10000)))
        pits[name] = randomized_pit(y, mu, CONFIG["pit_seed"] + 100*j + (0 if tag == "epilepsy" else 10000))[0]
    perf = pd.DataFrame(rows).sort_values("NLL").reset_index(drop=True)
    save_table(perf, f"08_{tag}_test_performance", f"Test performance: {design.name}")

    cal = calibration_table(test[design.y_col].to_numpy(float), {m: test[c].to_numpy(float) for m, c in pred_cols.items()}, n_bins=8)
    save_table(cal, f"09_{tag}_calibration", f"Calibration table: {design.name}")

    agg = group_aggregate_table(test, design.cluster_col, design.y_col, pred_cols)
    save_table(agg, f"10_{tag}_group_aggregates", f"Held-out biological-unit aggregate counts: {design.name}")

    boot_ci, paired = cluster_bootstrap_metrics(
        test, design.y_col, design.cluster_col, pred_cols,
        B=CONFIG["bootstrap_repetitions"], seed=SEED + (111 if tag == "epilepsy" else 222)
    )
    save_table(boot_ci, f"11_{tag}_cluster_bootstrap_ci", f"Cluster-bootstrap uncertainty: {design.name}")
    save_table(paired, f"12_{tag}_paired_bootstrap_neural", f"Paired cluster-bootstrap NLL differences versus Neural AIM-DSPP: {design.name}")

    return {
        "test": test,
        "pred_cols": pred_cols,
        "performance": perf,
        "calibration": cal,
        "aggregates": agg,
        "bootstrap": boot_ci,
        "paired": paired,
        "pits": pits,
    }


epil_res = evaluate_design(epil_design, epil_models, "epilepsy")
owl_res = evaluate_design(owl_design, owl_models, "owls")

# -----------------------------------------------------------------------------
# Permutation importance for the neural modulation score
# -----------------------------------------------------------------------------

def neural_permutation_importance(design: DatasetDesign, model: NeuralAIMModel, test: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = test[design.y_col].to_numpy(float)
    base_mu = model.predict_mu(test)
    base_nll = poisson_nll(y, base_mu)
    rows = []
    for col in design.neural_cols:
        deltas = []
        for _ in range(B):
            p = test.copy()
            p[col] = rng.permutation(p[col].to_numpy())
            nll = poisson_nll(y, model.predict_mu(p))
            deltas.append(nll - base_nll)
        vals = np.asarray(deltas)
        rows.append({
            "Feature": col,
            "BaseNLL": base_nll,
            "MeanDeltaNLL": vals.mean(),
            "MedianDeltaNLL": np.median(vals),
            "CI2.5": np.quantile(vals, 0.025),
            "CI97.5": np.quantile(vals, 0.975),
        })
    return pd.DataFrame(rows).sort_values("MeanDeltaNLL", ascending=False).reset_index(drop=True)


epil_perm = neural_permutation_importance(epil_design, epil_models["Neural AIM-DSPP"], epil_res["test"], CONFIG["permutation_repetitions"], SEED + 333)
owl_perm = neural_permutation_importance(owl_design, owl_models["Neural AIM-DSPP"], owl_res["test"], CONFIG["permutation_repetitions"], SEED + 444)
save_table(epil_perm, "13_epilepsy_neural_permutation_importance", "Neural AIM-DSPP permutation importance: epilepsy")
save_table(owl_perm, "14_owls_neural_permutation_importance", "Neural AIM-DSPP permutation importance: owls")

# -----------------------------------------------------------------------------
# Neural modulation effect grids
# -----------------------------------------------------------------------------

def epil_effect_grid(model: NeuralAIMModel, d: pd.DataFrame) -> pd.DataFrame:
    ages = np.linspace(d["age"].quantile(0.05), d["age"].quantile(0.95), 80)
    median_log_base = d["log_base"].median()
    rows = []
    for trt in [0, 1]:
        for period in [1, 2, 3, 4]:
            g = pd.DataFrame({
                "age": ages,
                "log_base": median_log_base,
                "period": period,
                "progabide": trt,
                "R": 1.0,
            })
            mod = model.centered_modulator(g)
            for a, m in zip(ages, mod):
                rows.append({"age": a, "period": period, "progabide": trt, "CenteredAIMMultiplier": m})
    return pd.DataFrame(rows)


def owl_effect_grid(model: NeuralAIMModel, d: pd.DataFrame) -> pd.DataFrame:
    arrivals = np.linspace(d["ArrivalTime"].quantile(0.03), d["ArrivalTime"].quantile(0.97), 100)
    median_log_brood = d["log_brood"].median()
    rows = []
    for deprived in [0, 1]:
        for male in [0, 1]:
            g = pd.DataFrame({
                "ArrivalTime": arrivals,
                "log_brood": median_log_brood,
                "deprived": deprived,
                "male_parent": male,
                "R": 1.0,
            })
            mod = model.centered_modulator(g)
            for a, m in zip(arrivals, mod):
                rows.append({"ArrivalTime": a, "deprived": deprived, "male_parent": male, "CenteredAIMMultiplier": m})
    return pd.DataFrame(rows)


epil_grid = epil_effect_grid(epil_models["Neural AIM-DSPP"], epil)
owl_grid = owl_effect_grid(owl_models["Neural AIM-DSPP"], owls)
epil_grid.to_csv(DAT / "epilepsy_neural_effect_grid.csv", index=False)
owl_grid.to_csv(DAT / "owls_neural_effect_grid.csv", index=False)

# -----------------------------------------------------------------------------
# Figures -- all displayed inline and saved as PDF + PNG
# -----------------------------------------------------------------------------

# Figure 1: Epilepsy data anatomy
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0,0].hist(epil["y"], bins=min(35, int(epil["y"].max()) + 1), edgecolor="black")
ax[0,0].set_xlabel("Two-week seizure count")
ax[0,0].set_ylabel("Frequency")
ax[0,0].set_title("Marginal seizure-count distribution")

subj = epil.groupby("subject", observed=True).agg(base=("base", "first"), followup=("y", "mean"), trt=("trt", "first"))
for trt, g in subj.groupby("trt", observed=True):
    ax[0,1].scatter(g["base"] / 4.0, g["followup"], alpha=0.75, label=str(trt))
mx = max((subj["base"] / 4.0).max(), subj["followup"].max())
ax[0,1].plot([0, mx], [0, mx], linestyle="--", linewidth=1)
ax[0,1].set_xlabel("Pre-treatment two-week baseline rate = base/4")
ax[0,1].set_ylabel("Mean follow-up count")
ax[0,1].set_title("External directing baseline versus follow-up")
ax[0,1].legend()

means = epil.groupby(["period", "trt"], observed=True)["y"].mean().reset_index()
for trt, g in means.groupby("trt", observed=True):
    ax[1,0].plot(g["period"], g["y"], marker="o", label=str(trt))
ax[1,0].set_xlabel("Two-week follow-up period")
ax[1,0].set_ylabel("Mean seizure count")
ax[1,0].set_title("Period-specific mean counts")
ax[1,0].legend()

heat = epil.pivot(index="subject", columns="period", values="y")
im = ax[1,1].imshow(heat.to_numpy(), aspect="auto")
ax[1,1].set_xlabel("Period")
ax[1,1].set_ylabel("Patient (ordered by identifier)")
ax[1,1].set_title("Longitudinal seizure-count heterogeneity")
ax[1,1].set_xticks(range(4), labels=[1,2,3,4])
fig.colorbar(im, ax=ax[1,1], label="Count")
save_figure(fig, "01_epilepsy_data_anatomy")

# Figure 2: Epilepsy model comparison
perf = epil_res["performance"].copy()
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
metrics = [("NLL", "Poisson negative log-likelihood"), ("PoissonDeviance", "Mean Poisson deviance"), ("RMSE", "RMSE"), ("PIT_KS", "Randomized-PIT KS statistic")]
for a, (m, lab) in zip(ax.ravel(), metrics):
    p = perf.sort_values(m)
    a.barh(p["Model"], p[m])
    a.set_xlabel(lab)
    a.set_title(lab)
save_figure(fig, "02_epilepsy_model_comparison")

# Figure 3: Epilepsy calibration and PIT
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
cal = epil_res["calibration"]
for model, g in cal.groupby("Model", observed=True):
    ax[0,0].plot(g["mean_predicted"], g["mean_observed"], marker="o", label=model)
maxcal = max(cal["mean_predicted"].max(), cal["mean_observed"].max())
ax[0,0].plot([0,maxcal],[0,maxcal], linestyle="--", linewidth=1)
ax[0,0].set_xlabel("Mean predicted count")
ax[0,0].set_ylabel("Mean observed count")
ax[0,0].set_title("Quantile-bin calibration")
ax[0,0].legend(fontsize=8)

for model, u in epil_res["pits"].items():
    xs = np.sort(u)
    ys = (np.arange(len(xs)) + 0.5) / len(xs)
    ax[0,1].plot(xs, ys, label=model)
ax[0,1].plot([0,1],[0,1], linestyle="--", linewidth=1)
ax[0,1].set_xlabel("Randomized PIT")
ax[0,1].set_ylabel("Empirical CDF")
ax[0,1].set_title("PIT empirical CDF")
ax[0,1].legend(fontsize=8)

neural_u = epil_res["pits"]["Neural AIM-DSPP"]
ax[1,0].hist(neural_u, bins=10, edgecolor="black")
ax[1,0].axhline(len(neural_u)/10, linestyle="--", linewidth=1)
ax[1,0].set_xlabel("Neural AIM-DSPP randomized PIT")
ax[1,0].set_ylabel("Frequency")
ax[1,0].set_title("Neural conditional-count calibration")

for model, col in epil_res["pred_cols"].items():
    mu = epil_res["test"][col].to_numpy(float)
    y = epil_res["test"]["y"].to_numpy(float)
    resid = (y-mu)/np.sqrt(np.clip(mu,1e-8,None))
    ax[1,1].scatter(mu, resid, s=22, alpha=0.55, label=model)
ax[1,1].axhline(0, linestyle="--", linewidth=1)
ax[1,1].set_xlabel("Predicted count")
ax[1,1].set_ylabel("Pearson residual")
ax[1,1].set_title("Conditional residual structure")
ax[1,1].legend(fontsize=8)
save_figure(fig, "03_epilepsy_calibration_and_residuals")

# Figure 4: Epilepsy patient-level aggregates
agg = epil_res["aggregates"]
models = list(epil_res["pred_cols"].keys())
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for a, model in zip(axes, models):
    a.scatter(agg["Observed"], agg[model], alpha=0.8)
    mx = max(agg["Observed"].max(), agg[model].max())
    a.plot([0,mx],[0,mx], linestyle="--", linewidth=1)
    rho = spearmanr(agg["Observed"], agg[model]).statistic
    a.set_title(f"{model}\nSpearman r={rho:.3f}")
    a.set_xlabel("Observed held-out patient total")
    a.set_ylabel("Predicted total")
for a in axes[len(models):]:
    a.axis("off")
save_figure(fig, "04_epilepsy_patient_aggregate_prediction")

# Figure 5: Epilepsy AI modulation and importance
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for (trt, period), g in epil_grid.groupby(["progabide", "period"], observed=True):
    if period in [1, 4]:
        label = f"{'Progabide' if trt else 'Placebo'}, period {period}"
        ax[0].plot(g["age"], g["CenteredAIMMultiplier"], label=label)
ax[0].axhline(1, linestyle="--", linewidth=1)
ax[0].set_xlabel("Age")
ax[0].set_ylabel("Centered neural AIM multiplier")
ax[0].set_title("Estimated nonlinear modulation at median baseline")
ax[0].legend(fontsize=8)

p = epil_perm.sort_values("MeanDeltaNLL")
ax[1].barh(p["Feature"], p["MeanDeltaNLL"])
ax[1].errorbar(p["MeanDeltaNLL"], p["Feature"],
               xerr=[p["MeanDeltaNLL"]-p["CI2.5"], p["CI97.5"]-p["MeanDeltaNLL"]],
               fmt="none", capsize=3)
ax[1].axvline(0, linestyle="--", linewidth=1)
ax[1].set_xlabel("Increase in held-out NLL after permutation")
ax[1].set_title("Neural feature importance")
save_figure(fig, "05_epilepsy_neural_modulation")

# Figure 6: Epilepsy training history + paired bootstrap
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
h = epil_tuning["neural_history"]
ax[0].plot(h["epoch"], h["ValidationNLL"], label="Validation NLL")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Validation NLL")
ax[0].set_title("Neural AIM-DSPP optimization")
ax[0].legend()
pb = epil_res["paired"].sort_values("MeanDifference")
ax[1].barh(pb["Benchmark"], pb["MeanDifference"])
ax[1].errorbar(pb["MeanDifference"], pb["Benchmark"],
               xerr=[pb["MeanDifference"]-pb["CI2.5"], pb["CI97.5"]-pb["MeanDifference"]],
               fmt="none", capsize=3)
ax[1].axvline(0, linestyle="--", linewidth=1)
ax[1].set_xlabel("Benchmark NLL - Neural NLL")
ax[1].set_title("Patient-cluster bootstrap comparison")
save_figure(fig, "06_epilepsy_training_and_bootstrap")

# Figure 7: Owl data anatomy
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0,0].hist(owls["y"], bins=min(40, int(owls["y"].max()) + 1), edgecolor="black")
ax[0,0].set_xlabel("Sibling-negotiation count")
ax[0,0].set_ylabel("Frequency")
ax[0,0].set_title("Marginal begging-count distribution")

grp = owls.groupby(["FoodTreatment", "SexParent"], observed=True)["y"].mean().reset_index()
labels = grp["FoodTreatment"].astype(str) + " / " + grp["SexParent"].astype(str)
ax[0,1].barh(labels, grp["y"])
ax[0,1].set_xlabel("Mean negotiation count")
ax[0,1].set_title("Food treatment and provisioning parent")

q = pd.qcut(owls["ArrivalTime"], q=12, duplicates="drop")
trend = owls.assign(arrival_bin=q).groupby("arrival_bin", observed=True).agg(arrival=("ArrivalTime","mean"), y=("y","mean")).reset_index()
ax[1,0].plot(trend["arrival"], trend["y"], marker="o")
ax[1,0].set_xlabel("Arrival time")
ax[1,0].set_ylabel("Mean negotiation count")
ax[1,0].set_title("Nonlinear time-of-arrival pattern")

ax[1,1].hist(nest_frailty["raw_history_rate"], bins=15, alpha=0.55, label="Raw history rate")
ax[1,1].hist(nest_frailty["shrunk_nest_rate"], bins=15, alpha=0.55, label="Shrunk directing rate")
ax[1,1].set_xlabel("Negotiations per chick-exposure unit")
ax[1,1].set_ylabel("Nests")
ax[1,1].set_title("History-only directing frailty")
ax[1,1].legend()
save_figure(fig, "07_owls_data_anatomy")

# Figure 8: Owls model comparison
perf = owl_res["performance"].copy()
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for a, (m, lab) in zip(ax.ravel(), metrics):
    p = perf.sort_values(m)
    a.barh(p["Model"], p[m])
    a.set_xlabel(lab)
    a.set_title(lab)
save_figure(fig, "08_owls_model_comparison")

# Figure 9: Owls calibration and PIT
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
cal = owl_res["calibration"]
for model, g in cal.groupby("Model", observed=True):
    ax[0,0].plot(g["mean_predicted"], g["mean_observed"], marker="o", label=model)
maxcal = max(cal["mean_predicted"].max(), cal["mean_observed"].max())
ax[0,0].plot([0,maxcal],[0,maxcal], linestyle="--", linewidth=1)
ax[0,0].set_xlabel("Mean predicted count")
ax[0,0].set_ylabel("Mean observed count")
ax[0,0].set_title("Quantile-bin calibration")
ax[0,0].legend(fontsize=8)

for model, u in owl_res["pits"].items():
    xs = np.sort(u)
    ys = (np.arange(len(xs)) + 0.5) / len(xs)
    ax[0,1].plot(xs, ys, label=model)
ax[0,1].plot([0,1],[0,1], linestyle="--", linewidth=1)
ax[0,1].set_xlabel("Randomized PIT")
ax[0,1].set_ylabel("Empirical CDF")
ax[0,1].set_title("PIT empirical CDF")
ax[0,1].legend(fontsize=8)

neural_u = owl_res["pits"]["Neural AIM-DSPP"]
ax[1,0].hist(neural_u, bins=10, edgecolor="black")
ax[1,0].axhline(len(neural_u)/10, linestyle="--", linewidth=1)
ax[1,0].set_xlabel("Neural AIM-DSPP randomized PIT")
ax[1,0].set_ylabel("Frequency")
ax[1,0].set_title("Neural conditional-count calibration")

for model, col in owl_res["pred_cols"].items():
    mu = owl_res["test"][col].to_numpy(float)
    y = owl_res["test"]["y"].to_numpy(float)
    resid = (y-mu)/np.sqrt(np.clip(mu,1e-8,None))
    ax[1,1].scatter(mu, resid, s=18, alpha=0.45, label=model)
ax[1,1].axhline(0, linestyle="--", linewidth=1)
ax[1,1].set_xlabel("Predicted count")
ax[1,1].set_ylabel("Pearson residual")
ax[1,1].set_title("Conditional residual structure")
ax[1,1].legend(fontsize=8)
save_figure(fig, "09_owls_calibration_and_residuals")

# Figure 10: Owl nest aggregate predictions
agg = owl_res["aggregates"]
models = list(owl_res["pred_cols"].keys())
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for a, model in zip(axes, models):
    a.scatter(agg["Observed"], agg[model], alpha=0.8)
    mx = max(agg["Observed"].max(), agg[model].max())
    a.plot([0,mx],[0,mx], linestyle="--", linewidth=1)
    rho = spearmanr(agg["Observed"], agg[model]).statistic
    a.set_title(f"{model}\nSpearman r={rho:.3f}")
    a.set_xlabel("Observed held-out nest total")
    a.set_ylabel("Predicted total")
for a in axes[len(models):]:
    a.axis("off")
save_figure(fig, "10_owls_nest_aggregate_prediction")

# Figure 11: Owl neural modulation and importance
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for (deprived, male), g in owl_grid.groupby(["deprived", "male_parent"], observed=True):
    label = f"{'Deprived' if deprived else 'Satiated'}, {'male' if male else 'female'} parent"
    ax[0].plot(g["ArrivalTime"], g["CenteredAIMMultiplier"], label=label)
ax[0].axhline(1, linestyle="--", linewidth=1)
ax[0].set_xlabel("Arrival time")
ax[0].set_ylabel("Centered neural AIM multiplier")
ax[0].set_title("Estimated behavioral modulation at median brood size")
ax[0].legend(fontsize=8)

p = owl_perm.sort_values("MeanDeltaNLL")
ax[1].barh(p["Feature"], p["MeanDeltaNLL"])
ax[1].errorbar(p["MeanDeltaNLL"], p["Feature"],
               xerr=[p["MeanDeltaNLL"]-p["CI2.5"], p["CI97.5"]-p["MeanDeltaNLL"]],
               fmt="none", capsize=3)
ax[1].axvline(0, linestyle="--", linewidth=1)
ax[1].set_xlabel("Increase in held-out NLL after permutation")
ax[1].set_title("Neural feature importance")
save_figure(fig, "11_owls_neural_modulation")

# Figure 12: Owl frailty diagnostics + paired bootstrap
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
analysis_rows = owls[owls["split"].isin(["train","validation","test"])].copy()
future_rate = analysis_rows.groupby("Nest", observed=True).agg(
    future_calls=("y", "sum"), future_exposure=("BroodSize", "sum")
).reset_index()
future_rate["future_rate"] = future_rate["future_calls"] / future_rate["future_exposure"]
future_rate = future_rate[["Nest", "future_rate"]]
fdiag = nest_frailty.merge(future_rate, on="Nest", how="left")
ax[0].scatter(fdiag["shrunk_nest_rate"], fdiag["future_rate"], alpha=0.8)
mx = np.nanmax([fdiag["shrunk_nest_rate"].max(), fdiag["future_rate"].max()])
ax[0].plot([0,mx],[0,mx], linestyle="--", linewidth=1)
rho = spearmanr(fdiag["shrunk_nest_rate"], fdiag["future_rate"], nan_policy="omit").statistic
ax[0].set_xlabel("History-only shrunk nest rate")
ax[0].set_ylabel("Rate in non-history observations")
ax[0].set_title(f"Directing-frailty persistence; Spearman r={rho:.3f}")

pb = owl_res["paired"].sort_values("MeanDifference")
ax[1].barh(pb["Benchmark"], pb["MeanDifference"])
ax[1].errorbar(pb["MeanDifference"], pb["Benchmark"],
               xerr=[pb["MeanDifference"]-pb["CI2.5"], pb["CI97.5"]-pb["MeanDifference"]],
               fmt="none", capsize=3)
ax[1].axvline(0, linestyle="--", linewidth=1)
ax[1].set_xlabel("Benchmark NLL - Neural NLL")
ax[1].set_title("Nest-cluster bootstrap comparison")
save_figure(fig, "12_owls_frailty_and_bootstrap")

# Figure 13: Cross-dataset synthesis
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for a, res, title in [(ax[0], epil_res, "Epilepsy"), (ax[1], owl_res, "Owls")]:
    p = res["performance"].copy()
    base_nll = float(p.loc[p["Model"] == "Directing baseline only", "NLL"].iloc[0])
    p["RelativeNLLImprovementPct"] = 100.0 * (base_nll - p["NLL"]) / base_nll
    p = p.sort_values("RelativeNLLImprovementPct")
    a.barh(p["Model"], p["RelativeNLLImprovementPct"])
    a.axvline(0, linestyle="--", linewidth=1)
    a.set_xlabel("NLL improvement over directing baseline (%)")
    a.set_title(title)
save_figure(fig, "13_cross_dataset_predictive_synthesis")

# -----------------------------------------------------------------------------
# Cross-dataset numerical summary
# -----------------------------------------------------------------------------

def cross_rows(tag: str, perf: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    baseline_nll = float(perf.loc[perf["Model"] == "Directing baseline only", "NLL"].iloc[0])
    for _, r in perf.iterrows():
        out.append({
            "Dataset": tag,
            "Model": r["Model"],
            "NLL": r["NLL"],
            "PoissonDeviance": r["PoissonDeviance"],
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "PIT_KS": r["PIT_KS"],
            "PIT_p": r["PIT_p"],
            "NLLImprovementOverBaselinePct": 100*(baseline_nll-r["NLL"])/baseline_nll,
        })
    return out

cross = pd.DataFrame(cross_rows("Epilepsy", epil_res["performance"]) + cross_rows("Owls", owl_res["performance"]))
save_table(cross, "15_cross_dataset_performance", "Cross-dataset empirical performance synthesis")

neural_norm = pd.DataFrame([
    {
        "Dataset": "Epilepsy",
        "TrainMeanLogRatioCenter": epil_models["Neural AIM-DSPP"].center,
        "GlobalScaleExpCenter": epil_models["Neural AIM-DSPP"].global_scale,
        "CenteredMultiplierGeometricMeanOnTrain": float(np.exp(np.mean(np.log(epil_models["Neural AIM-DSPP"].centered_modulator(epil[epil["split"]=="train"]))))),
    },
    {
        "Dataset": "Owls",
        "TrainMeanLogRatioCenter": owl_models["Neural AIM-DSPP"].center,
        "GlobalScaleExpCenter": owl_models["Neural AIM-DSPP"].global_scale,
        "CenteredMultiplierGeometricMeanOnTrain": float(np.exp(np.mean(np.log(owl_models["Neural AIM-DSPP"].centered_modulator(owls[owls["split"]=="train"]))))),
    },
])
save_table(neural_norm, "16_neural_identifiable_normalization", "Post-fit geometric normalization of the neural AIM representation")

# -----------------------------------------------------------------------------
# Automatically generated result note (not intended to replace final manuscript
# interpretation; it records actual run values so the later LaTeX write-up can
# be grounded in the downloaded output rather than memory).
# -----------------------------------------------------------------------------

def best_model_name(perf: pd.DataFrame) -> str:
    return str(perf.sort_values("NLL").iloc[0]["Model"])


def get_perf(perf: pd.DataFrame, model: str, col: str) -> float:
    return float(perf.loc[perf["Model"] == model, col].iloc[0])


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")

lines = []
lines.append(r"\section{Auto-generated empirical-results note}")
lines.append(r"\textit{This file is generated directly from the Python run and is intended as a numerical audit trail for later manuscript writing.}")
lines.append("")
lines.append(r"\subsection{Epilepsy application}")
lines.append(
    "The MASS \\texttt{epil} analysis used %d patients and %d two-week follow-up counts. "
    "The directing baseline was the pre-treatment eight-week seizure count divided by four. "
    "Patients were separated across train, validation, and test sets." % (epil["subject"].nunique(), len(epil))
)
for m in epil_res["performance"]["Model"]:
    lines.append(
        f"\\noindent {tex_escape(m)}: NLL={get_perf(epil_res['performance'],m,'NLL'):.4f}, "
        f"deviance={get_perf(epil_res['performance'],m,'PoissonDeviance'):.4f}, "
        f"RMSE={get_perf(epil_res['performance'],m,'RMSE'):.4f}, "
        f"PIT KS p={get_perf(epil_res['performance'],m,'PIT_p'):.4g}.\\\\"
    )
lines.append("")
lines.append(r"\subsection{Barn-owl application}")
lines.append(
    "The glmmTMB \\texttt{Owls} analysis used %d nests and %d repeated begging-count observations. "
    "A history-only Gamma--Poisson-shrunk nest rate was multiplied by brood size to form the directing baseline for each evaluated bout. "
    "History observations were excluded from model training and test scoring." % (owls["Nest"].nunique(), len(owls))
)
for m in owl_res["performance"]["Model"]:
    lines.append(
        f"\\noindent {tex_escape(m)}: NLL={get_perf(owl_res['performance'],m,'NLL'):.4f}, "
        f"deviance={get_perf(owl_res['performance'],m,'PoissonDeviance'):.4f}, "
        f"RMSE={get_perf(owl_res['performance'],m,'RMSE'):.4f}, "
        f"PIT KS p={get_perf(owl_res['performance'],m,'PIT_p'):.4g}.\\\\"
    )
lines.append("")
lines.append(r"\subsection{Audit statement}")
lines.append(
    "The fixed benchmark set consisted of a directing-baseline-only model, a linear Poisson AIM, a spline Poisson AIM, "
    "Poisson gradient boosting, and Neural AIM-DSPP. Hyperparameters were selected using validation data only. "
    "No model was added to or removed from the test comparison on the basis of its test performance."
)
with open(OUT / "empirical_results_auto.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# -----------------------------------------------------------------------------
# Machine-readable summary
# -----------------------------------------------------------------------------
summary = {
    "datasets": {
        "epilepsy": {
            "rows": int(len(epil)),
            "patients": int(epil["subject"].nunique()),
            "loading_route": epil_route,
            "source_doi": "10.2307/2532086",
            "best_test_model_by_nll": best_model_name(epil_res["performance"]),
            "performance": epil_res["performance"].to_dict(orient="records"),
            "neural_global_scale": epil_models["Neural AIM-DSPP"].global_scale,
        },
        "owls": {
            "rows": int(len(owls)),
            "nests": int(owls["Nest"].nunique()),
            "loading_route": owls_route,
            "source_doi": "10.1016/j.anbehav.2007.01.027",
            "history_global_rate": float(owls.attrs.get("global_history_rate", np.nan)),
            "best_test_model_by_nll": best_model_name(owl_res["performance"]),
            "performance": owl_res["performance"].to_dict(orient="records"),
            "neural_global_scale": owl_models["Neural AIM-DSPP"].global_scale,
        },
    },
    "scientific_guardrails": {
        "event_times_available": False,
        "time_rescaling_used": False,
        "randomized_pit_used": True,
        "epilepsy_split_by_patient": True,
        "owl_test_outcomes_used_for_frailty_estimation": False,
        "benchmark_selection_conditional_on_test_performance": False,
    },
}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# -----------------------------------------------------------------------------
# README
# -----------------------------------------------------------------------------
readme = f"""
AIM-DSPP REAL BIOLOGICAL DATA OUTPUTS
=====================================

Generated by: aim_dspp_real_biology_colab.py
Seed: {SEED}
Device: {DEVICE}

DATASET 1: MASS::epil
---------------------
Biological response: epileptic seizure counts in four successive two-week periods.
Rows: {len(epil)}
Patients: {epil['subject'].nunique()}
AIM-DSPP directing baseline: pre-treatment 8-week seizure count divided by 4.
Split: patient-level train/validation/test; patients do not cross splits.
Primary statistical source: Thall & Vail (1990), Biometrics, DOI 10.2307/2532086.

DATASET 2: glmmTMB::Owls
------------------------
Biological response: sibling-negotiation / begging-call counts by barn-owl nestlings.
Rows: {len(owls)}
Nests: {owls['Nest'].nunique()}
AIM-DSPP directing baseline: history-only Gamma-Poisson-shrunk nest rate times brood size.
History responses are used only to estimate nest frailty and are excluded from model fitting/test scoring.
Primary biological source: Roulin & Bersier (2007), Animal Behaviour,
DOI 10.1016/j.anbehav.2007.01.027.

FIXED MODEL SET
---------------
1. Directing baseline only
2. Linear Poisson AIM
3. Spline Poisson AIM
4. Poisson Gradient Boosting
5. Neural AIM-DSPP

IMPORTANT INTERPRETATION
------------------------
These datasets provide interval counts rather than exact event timestamps. The empirical analysis
therefore assesses conditional Poisson calibration with randomized PIT residuals. It does NOT apply
the continuous-time random-time-rescaling theorem to pseudo event times.

OUTPUT CONTENTS
---------------
figures/   : all figures in PDF and PNG
 tables/    : substantive tables in CSV and LaTeX
 data/      : prepared datasets, held-out predictions, and effect grids
 models/    : fitted sklearn objects and neural state dictionaries
 metadata/  : dataset documentation, configuration, software versions
 summary.json
 empirical_results_auto.tex
 README.txt

No benchmark was selected or removed on the basis of test performance.
"""
with open(OUT / "README.txt", "w", encoding="utf-8") as f:
    f.write(readme)

# -----------------------------------------------------------------------------
# Zip and Colab download
# -----------------------------------------------------------------------------
zip_base = str(OUT)
zip_path = shutil.make_archive(zip_base, "zip", root_dir=OUT)

print("\n" + "=" * 96)
print("REAL BIOLOGICAL DATA ANALYSIS COMPLETE")
print("=" * 96)
print("Output folder:", OUT)
print("ZIP archive:  ", zip_path)
print("All substantive tables were printed inline and all figures were displayed inline.")
print("Please send the resulting ZIP back for the manuscript empirical-application section.")

try:
    from google.colab import files
    print("Triggering Colab download of the ZIP ...")
    files.download(zip_path)
except Exception as e:
    print("Automatic Colab download not available in this environment.")
    print("Download manually from:", zip_path)
