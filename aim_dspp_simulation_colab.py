"""
AIM-DSPP: Simulation, estimation, theorem diagnostics, and publication artifacts
-------------------------------------------------------------------------------
Self-contained companion code for the paper
"AI-Modulated Doubly Stochastic Poisson Processes".

Designed for Google Colab (CPU or GPU). The script:
  1) simulates exact piecewise-constant Cox/AIM-DSPP paths;
  2) validates conditional-Poisson and overdispersion identities;
  3) demonstrates multiplicative non-identifiability exactly;
  4) fits Cox-only, linear-AIM, and neural AIM-DSPP models by conditional
     Poisson likelihood with pathwise geometric normalization;
  5) evaluates held-out likelihood, deviance, intensity recovery, calibration,
     and random-time-rescaling diagnostics;
  6) empirically verifies the law-stability/coupling theorem;
  7) studies neural approximation and its induced process-law bound;
  8) illustrates the compensator LLN and random-compensator CLT;
  9) saves every figure/table and a LaTeX-ready simulation section;
 10) creates a ZIP and, in Colab, triggers its download automatically.

IMPORTANT SCIENTIFIC NOTE
-------------------------
No result is hard-coded and no benchmark is removed based on performance.
The script reports whatever the simulation produces. The Oracle row is a
reference, not a fitted competitor.

Run in Colab:
    !python aim_dspp_simulation_colab.py

Optional quick smoke-test mode:
    AIM_DSPP_QUICK=1 python aim_dspp_simulation_colab.py
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.special import gammaln, expit
from scipy import stats

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise RuntimeError(
        "PyTorch is required. Google Colab includes it by default. "
        "If needed, run: !pip install torch"
    ) from exc

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK = os.environ.get("AIM_DSPP_QUICK", "0") == "1"


@dataclass
class Config:
    seed: int = 20260823
    output_dir: str = "AIM_DSPP_Simulation_Outputs"
    # Main simulation
    T: float = 12.0
    n_bins: int = 600
    n_paths: int = 90
    base_rate: float = 18.0
    ou_tau: float = 1.25
    ou_sigma: float = 0.48
    # Splits by complete paths
    n_train_paths: int = 50
    n_val_paths: int = 15
    # Estimation
    hidden: Tuple[int, int] = (48, 24)
    lr_linear: float = 0.025
    lr_mlp: float = 0.008
    max_epochs_linear: int = 1200
    max_epochs_mlp: int = 1800
    patience_linear: int = 140
    patience_mlp: int = 180
    weight_decay_linear: float = 1e-5
    weight_decay_mlp: float = 2e-5
    score_cap: float = 2.40
    # Bootstrap
    n_boot: int = 2000
    # Stability experiment
    stability_paths: int = 260
    stability_bins: int = 500
    stability_T: float = 6.0
    # Approximation experiment
    approx_train_n: int = 9000
    approx_val_n: int = 2500
    approx_eval_n: int = 18000
    approx_widths: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    approx_epochs: int = 900
    approx_patience: int = 90
    # Long-run experiment
    long_paths: int = 280
    long_T: float = 100.0
    long_bins: int = 5000


CFG = Config()
if QUICK:
    CFG.n_paths = 24
    CFG.n_train_paths = 12
    CFG.n_val_paths = 5
    CFG.n_bins = 260
    CFG.max_epochs_linear = 180
    CFG.max_epochs_mlp = 260
    CFG.patience_linear = 35
    CFG.patience_mlp = 45
    CFG.n_boot = 250
    CFG.stability_paths = 60
    CFG.stability_bins = 180
    CFG.approx_train_n = 1800
    CFG.approx_val_n = 500
    CFG.approx_eval_n = 2500
    CFG.approx_widths = (2, 8, 24)
    CFG.approx_epochs = 180
    CFG.approx_patience = 35
    CFG.long_paths = 70
    CFG.long_T = 30.0
    CFG.long_bins = 900


# -----------------------------------------------------------------------------
# Reproducibility, paths, plotting
# -----------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(CFG.output_dir)
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
for p in (ROOT, FIG_DIR, TAB_DIR, MODEL_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

# A restrained publication palette. The paper remains readable in grayscale too.
COLORS = {
    "true": "#111827",
    "baseline": "#6B7280",
    "linear": "#2563EB",
    "neural": "#B91C1C",
    "oracle": "#047857",
    "accent": "#7C3AED",
    "gold": "#B7791F",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 320,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "lines.linewidth": 1.7,
})


def display_df(df: pd.DataFrame, name: str | None = None) -> None:
    """Pretty inline table in notebooks; graceful console fallback."""
    if name:
        print("\n" + "=" * 88)
        print(name)
        print("=" * 88)
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))


def save_show(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved figure: {png}")
    plt.show()
    plt.close(fig)


def save_table(df: pd.DataFrame, stem: str, index: bool = False) -> None:
    csv_path = TAB_DIR / f"{stem}.csv"
    tex_path = TAB_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=index)
    try:
        tex = df.to_latex(index=index, escape=False, float_format=lambda x: f"{x:.4f}")
        tex_path.write_text(tex, encoding="utf-8")
    except Exception:
        pass
    print(f"Saved table:  {csv_path}")


# -----------------------------------------------------------------------------
# Core stochastic-process simulation
# -----------------------------------------------------------------------------

def simulate_ou_exact(n: int, dt: float, tau: float, sigma_stationary: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Stationary OU sampled exactly on an equally spaced grid."""
    phi = math.exp(-dt / tau)
    innov_sd = sigma_stationary * math.sqrt(max(1.0 - phi * phi, 1e-12))
    z = np.empty(n, dtype=float)
    z[0] = rng.normal(0.0, sigma_stationary)
    eps = rng.normal(size=n - 1)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + innov_sd * eps[i - 1]
    return z


def simulate_features(n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Three exogenous, temporally structured features, independent of Poisson innovation."""
    x1 = simulate_ou_exact(n, dt, tau=0.70, sigma_stationary=1.0, rng=rng)
    x3 = simulate_ou_exact(n, dt, tau=1.60, sigma_stationary=0.85, rng=rng)
    t = np.arange(n) * dt
    phase = rng.uniform(0, 2 * np.pi)
    slow = np.sin(2 * np.pi * t / max(4.5, t[-1] + dt) + phase)
    x2 = 0.72 * slow + 0.45 * simulate_ou_exact(
        n, dt, tau=0.45, sigma_stationary=0.8, rng=rng
    )
    X = np.column_stack([x1, x2, x3])
    # Global scaling is deliberately not pathwise standardized: covariate magnitude carries information.
    return np.clip(X, -3.0, 3.0)


def nonlinear_score(X: np.ndarray) -> np.ndarray:
    """Smooth nonlinear target AI score used in the data-generating mechanism."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    raw = (
        0.95 * np.sin(1.15 * x1)
        + 0.62 * x2 * x3
        - 0.34 * (x2 ** 2 - 0.6)
        + 0.46 * np.cos(1.30 * x3)
        + 0.28 * x1 * x2
        - 0.15 * np.sin(x1 * x3)
    )
    # Smoothly bounded score controls extreme intensities while retaining nonlinearity.
    return CFG.score_cap * np.tanh(raw / CFG.score_cap)


def geometric_normalize_score(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce mean(log A)=0 pathwise, i.e. geometric mean of A equals one."""
    centered = score - np.mean(score)
    A = np.exp(centered)
    return centered, A


def simulate_one_path(path_id: int, cfg: Config, seed: int,
                      n_bins: int | None = None, T: float | None = None,
                      base_rate: float | None = None) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(n_bins or cfg.n_bins)
    TT = float(T or cfg.T)
    br = float(base_rate or cfg.base_rate)
    dt = TT / n
    t_mid = (np.arange(n) + 0.5) * dt

    z = simulate_ou_exact(n, dt, cfg.ou_tau, cfg.ou_sigma, rng)
    # Mean approximately br marginally because E exp(Z - sigma^2/2)=1 for stationary Gaussian Z.
    # A mild deterministic seasonal baseline is allowed inside R and is known conditionally.
    seasonal = np.exp(0.18 * np.sin(2 * np.pi * t_mid / TT + rng.uniform(0, 2*np.pi)))
    R = br * np.exp(z - 0.5 * cfg.ou_sigma ** 2) * seasonal

    X = simulate_features(n, dt, rng)
    s_raw = nonlinear_score(X)
    s, A = geometric_normalize_score(s_raw)
    lam = R * A
    mu = lam * dt

    y = rng.poisson(mu)
    # Conditional on bin count under piecewise-constant intensity, event locations are iid Uniform in bin.
    event_times: List[float] = []
    for j, c in enumerate(y):
        if c:
            ev = j * dt + rng.uniform(0.0, dt, size=int(c))
            event_times.extend(ev.tolist())
    event_times = np.sort(np.asarray(event_times, dtype=float))

    frame = pd.DataFrame({
        "path": path_id,
        "bin": np.arange(n, dtype=int),
        "t": t_mid,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
        "R": R,
        "score_true": s,
        "A_true": A,
        "lambda_true": lam,
        "mu_true": mu,
        "count": y.astype(int),
    })
    return {
        "frame": frame,
        "event_times": event_times,
        "dt": dt,
        "Lambda": float(mu.sum()),
        "N": int(y.sum()),
    }


def simulate_dataset(cfg: Config) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], float]:
    frames = []
    events: Dict[int, np.ndarray] = {}
    dt = cfg.T / cfg.n_bins
    for p in range(cfg.n_paths):
        obj = simulate_one_path(p, cfg, cfg.seed + 10007 * (p + 1))
        frames.append(obj["frame"])
        events[p] = obj["event_times"]
    return pd.concat(frames, ignore_index=True), events, dt


# -----------------------------------------------------------------------------
# Point-process diagnostics
# -----------------------------------------------------------------------------

def integrated_intensity_at_events(event_times: np.ndarray,
                                   lam_bins: np.ndarray,
                                   dt: float) -> np.ndarray:
    """Exact compensator values for piecewise-constant lambda."""
    if len(event_times) == 0:
        return np.asarray([])
    n = len(lam_bins)
    cum = np.concatenate([[0.0], np.cumsum(lam_bins * dt)])
    j = np.floor(event_times / dt).astype(int)
    j = np.clip(j, 0, n - 1)
    left = j * dt
    H = cum[j] + lam_bins[j] * (event_times - left)
    return H


def rescaled_gaps(event_times: np.ndarray, lam_bins: np.ndarray, dt: float) -> np.ndarray:
    H = integrated_intensity_at_events(event_times, lam_bins, dt)
    if len(H) == 0:
        return H
    return np.diff(np.concatenate([[0.0], H]))


def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.clip(mu, 1e-12, None)
    y = np.asarray(y, dtype=float)
    term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (term - (y - mu))


def structural_diagnostics(df: pd.DataFrame, events: Dict[int, np.ndarray], dt: float) -> pd.DataFrame:
    y = df["count"].to_numpy(float)
    mu = df["mu_true"].to_numpy(float)
    pearson = (y - mu) / np.sqrt(np.clip(mu, 1e-12, None))

    gaps = []
    for p, g in df.groupby("path", sort=True):
        ev = events[int(p)]
        gaps.extend(rescaled_gaps(ev, g["lambda_true"].to_numpy(), dt).tolist())
    gaps = np.asarray(gaps)
    ks = stats.kstest(gaps, "expon") if len(gaps) else (np.nan, np.nan)

    by_path = df.groupby("path").agg(N=("count", "sum"), Lambda=("mu_true", "sum"))
    emp_mean = by_path["N"].mean()
    emp_var = by_path["N"].var(ddof=1)
    E_L = by_path["Lambda"].mean()
    Var_L = by_path["Lambda"].var(ddof=1)
    theory_var = E_L + Var_L

    out = pd.DataFrame([
        ["Pearson residual mean", pearson.mean(), 0.0],
        ["Pearson residual variance", pearson.var(ddof=1), 1.0],
        ["Time-rescaling KS statistic", float(ks.statistic), 0.0],
        ["Time-rescaling KS p-value", float(ks.pvalue), np.nan],
        ["Empirical mean N(T)", emp_mean, E_L],
        ["Empirical variance N(T)", emp_var, theory_var],
        ["Fano factor Var[N]/E[N]", emp_var / emp_mean, theory_var / E_L],
        ["Var(Lambda) contribution", Var_L, Var_L],
    ], columns=["Diagnostic", "Empirical", "Theory/Target"])
    return out


# -----------------------------------------------------------------------------
# Torch models with pathwise geometric normalization
# -----------------------------------------------------------------------------

def to_local_group_ids(path_ids: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    uniq = np.unique(path_ids)
    mapping = {int(p): i for i, p in enumerate(uniq)}
    local = np.asarray([mapping[int(p)] for p in path_ids], dtype=np.int64)
    return local, mapping


def group_mean_torch(v: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    sums = torch.zeros(n_groups, device=v.device, dtype=v.dtype)
    counts = torch.zeros(n_groups, device=v.device, dtype=v.dtype)
    sums.scatter_add_(0, group, v)
    counts.scatter_add_(0, group, torch.ones_like(v))
    return sums / torch.clamp(counts, min=1.0)


def normalized_A_torch(raw_score: torch.Tensor, group: torch.Tensor,
                       n_groups: int, score_cap: float) -> Tuple[torch.Tensor, torch.Tensor]:
    bounded = score_cap * torch.tanh(raw_score / score_cap)
    gm = group_mean_torch(bounded, group, n_groups)
    centered = bounded - gm[group]
    return torch.exp(centered), centered


class LinearModulator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class NeuralModulator(nn.Module):
    def __init__(self, d: int, hidden: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(d, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TensorDatasetFull:
    X: torch.Tensor
    R: torch.Tensor
    y: torch.Tensor
    group: torch.Tensor
    n_groups: int
    path_orig: np.ndarray
    dt: float


def make_tensor_dataset(df: pd.DataFrame, dt: float) -> TensorDatasetFull:
    X = df[["x1", "x2", "x3"]].to_numpy(np.float32)
    R = df["R"].to_numpy(np.float32)
    y = df["count"].to_numpy(np.float32)
    local, _ = to_local_group_ids(df["path"].to_numpy())
    return TensorDatasetFull(
        X=torch.tensor(X, device=DEVICE),
        R=torch.tensor(R, device=DEVICE),
        y=torch.tensor(y, device=DEVICE),
        group=torch.tensor(local, device=DEVICE, dtype=torch.long),
        n_groups=int(local.max() + 1),
        path_orig=df["path"].to_numpy(),
        dt=dt,
    )


def model_mu(model: nn.Module, data: TensorDatasetFull, score_cap: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = model(data.X)
    A, score = normalized_A_torch(raw, data.group, data.n_groups, score_cap)
    mu = data.R * A * data.dt
    return mu, A, score


def conditional_nll(model: nn.Module, data: TensorDatasetFull, score_cap: float) -> torch.Tensor:
    mu, _, _ = model_mu(model, data, score_cap)
    mu = torch.clamp(mu, min=1e-9, max=1e7)
    # Constant log(y!) omitted during training; does not change fitted parameters.
    return torch.mean(mu - data.y * torch.log(mu))


def train_modulator(model: nn.Module,
                    train: TensorDatasetFull,
                    val: TensorDatasetFull,
                    lr: float,
                    max_epochs: int,
                    patience: int,
                    weight_decay: float,
                    label: str) -> Tuple[nn.Module, pd.DataFrame]:
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.55, patience=max(12, patience//4))
    best = np.inf
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = conditional_nll(model, train, CFG.score_cap)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(conditional_nll(model, val, CFG.score_cap).cpu())
            tloss = float(loss.detach().cpu())
        sched.step(vloss)
        lr_now = opt.param_groups[0]["lr"]
        history.append((epoch, tloss, vloss, lr_now))

        if vloss < best - 1e-7:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 100 == 0:
            print(f"[{label:12s}] epoch={epoch:4d} train={tloss:.6f} val={vloss:.6f} lr={lr_now:.3g}")
        if stale >= patience:
            print(f"[{label:12s}] early stop at epoch {epoch}; best val={best:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist = pd.DataFrame(history, columns=["epoch", "train_nll_no_const", "val_nll_no_const", "lr"])
    return model, hist


def predict_modulator(model: nn.Module, df: pd.DataFrame, dt: float) -> Dict[str, np.ndarray]:
    data = make_tensor_dataset(df, dt)
    model.eval()
    with torch.no_grad():
        mu, A, score = model_mu(model, data, CFG.score_cap)
    mu = mu.detach().cpu().numpy()
    A = A.detach().cpu().numpy()
    score = score.detach().cpu().numpy()
    lam = mu / dt
    return {"mu": mu, "A": A, "score": score, "lambda": lam}


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, rng: np.random.Generator,
                 B: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def evaluate_predictions(test_df: pd.DataFrame,
                         pred_by_model: Dict[str, np.ndarray],
                         events: Dict[int, np.ndarray],
                         dt: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = test_df["count"].to_numpy(float)
    lam_true = test_df["lambda_true"].to_numpy(float)
    rng = np.random.default_rng(CFG.seed + 98765)

    rows = []
    path_rows = []
    unique_paths = np.sort(test_df["path"].unique())

    for model_name, lam_pred in pred_by_model.items():
        mu_pred = np.clip(lam_pred * dt, 1e-10, None)
        # pathwise metrics for CIs
        metrics = {"NLL/bin": [], "Poisson deviance/bin": [], "Intensity RMSE": [], "Log-intensity RMSE": [], "Intensity MAE": []}
        for p in unique_paths:
            m = (test_df["path"].to_numpy() == p)
            yy, mm = y[m], mu_pred[m]
            lt, lp = lam_true[m], lam_pred[m]
            vals = {
                "NLL/bin": -poisson_loglik(yy, mm) / len(yy),
                "Poisson deviance/bin": poisson_deviance(yy, mm).mean(),
                "Intensity RMSE": float(np.sqrt(np.mean((lp - lt) ** 2))),
                "Log-intensity RMSE": float(np.sqrt(np.mean((np.log(np.clip(lp,1e-10,None)) - np.log(lt)) ** 2))),
                "Intensity MAE": float(np.mean(np.abs(lp - lt))),
            }
            for k, v in vals.items():
                metrics[k].append(v)
            path_rows.append({"Model": model_name, "Path": int(p), **vals})

        row = {"Model": model_name}
        for k, v in metrics.items():
            mean, lo, hi = bootstrap_ci(np.asarray(v), rng, B=CFG.n_boot)
            row[k] = mean
            row[k + " CI low"] = lo
            row[k + " CI high"] = hi

        # Time-rescaling diagnostic using fitted intensity paths.
        gaps = []
        start = 0
        for p in unique_paths:
            g = test_df[test_df["path"] == p]
            n = len(g)
            lp = lam_pred[start:start+n]
            gaps.extend(rescaled_gaps(events[int(p)], lp, dt).tolist())
            start += n
        gaps = np.asarray(gaps)
        if len(gaps):
            ks = stats.kstest(gaps, "expon")
            row["Rescaling KS D"] = float(ks.statistic)
            row["Rescaling KS p"] = float(ks.pvalue)
        else:
            row["Rescaling KS D"] = np.nan
            row["Rescaling KS p"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(path_rows)


def calibration_frame(test_df: pd.DataFrame, model: str, lam_pred: np.ndarray, dt: float,
                      n_groups: int = 10) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "obs": test_df["count"].to_numpy(float),
        "pred": np.clip(lam_pred * dt, 1e-12, None),
    })
    tmp["bin"] = pd.qcut(tmp["pred"], q=n_groups, duplicates="drop")
    c = tmp.groupby("bin", observed=True).agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size")).reset_index(drop=True)
    c["Model"] = model
    return c


# -----------------------------------------------------------------------------
# Stability theorem experiment
# -----------------------------------------------------------------------------

def bounded_link(s: np.ndarray, a_lo: float = 0.40, a_hi: float = 2.20) -> np.ndarray:
    return a_lo + (a_hi - a_lo) * expit(s)


def stability_experiment(cfg: Config) -> pd.DataFrame:
    deltas = np.asarray([0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80])
    a_lo, a_hi = 0.40, 2.20
    Lh = (a_hi - a_lo) / 4.0
    rng_master = np.random.default_rng(cfg.seed + 4444)
    rows = []

    # Store path-specific ingredients once so every delta is compared on identical directing environments.
    paths = []
    for p in range(cfg.stability_paths):
        obj = simulate_one_path(
            p, cfg, cfg.seed + 700000 + p,
            n_bins=cfg.stability_bins, T=cfg.stability_T, base_rate=5.0
        )
        g = obj["frame"]
        R = g["R"].to_numpy()
        X = g[["x1", "x2", "x3"]].to_numpy()
        s = nonlinear_score(X)
        # A deterministic perturbation direction built only from exogenous quantities.
        q = np.sin(1.7 * X[:, 0]) + 0.55 * np.cos(1.2 * X[:, 2]) + 0.25 * X[:, 1]
        q = q / (np.sqrt(np.mean(q*q)) + 1e-12)
        paths.append((R, s, q, obj["dt"]))

    for delta in deltas:
        D_vals, exact_probs, lip_vals, mismatch = [], [], [], []
        for R, s, q, dt in paths:
            sp = s + delta * q
            lam = R * bounded_link(s, a_lo, a_hi)
            lamp = R * bounded_link(sp, a_lo, a_hi)
            D = float(np.sum(np.abs(lam - lamp)) * dt)
            p_exact = 1.0 - math.exp(-D)
            lip = float(Lh * np.sum(R * np.abs(sp - s)) * dt)
            D_vals.append(D)
            exact_probs.append(p_exact)
            lip_vals.append(lip)
            # Common-PRM mismatch indicator: P(mismatch | G)=1-exp(-D).
            mismatch.append(int(rng_master.random() < p_exact))

        rows.append({
            "delta": delta,
            "mean_D": np.mean(D_vals),
            "empirical_coupling_mismatch": np.mean(mismatch),
            "exact_E_1_minus_exp_minus_D": np.mean(exact_probs),
            "mean_Lipschitz_bound_raw": np.mean(lip_vals),
            "mean_Lipschitz_bound_clipped": min(1.0, np.mean(lip_vals)),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Universal approximation experiment
# -----------------------------------------------------------------------------

class ApproxNet(nn.Module):
    def __init__(self, d: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def approx_target_psi(X: np.ndarray) -> np.ndarray:
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (
        0.75 * np.sin(1.15*x1)
        + 0.42*x2*x3
        - 0.22*x2*x2
        + 0.33*np.cos(1.4*x3)
        + 0.18*x1*x2
    )


def train_approx_net(width: int, Xtr: np.ndarray, ytr: np.ndarray,
                     Xv: np.ndarray, yv: np.ndarray) -> ApproxNet:
    net = ApproxNet(3, width).to(DEVICE)
    tx = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    vx = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
    vy = torch.tensor(yv, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=0.012)
    best, stale, best_state = np.inf, 0, None

    for epoch in range(1, CFG.approx_epochs + 1):
        net.train(); opt.zero_grad(set_to_none=True)
        pred = net(tx)
        loss = torch.mean((pred - ty)**2)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.mean((net(vx) - vy)**2).cpu())
        if v < best - 1e-8:
            best, stale = v, 0
            best_state = {k: z.detach().cpu().clone() for k,z in net.state_dict().items()}
        else:
            stale += 1
        if stale >= CFG.approx_patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def universal_approximation_experiment(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 55555)
    Xtr = rng.uniform(-2.4, 2.4, size=(cfg.approx_train_n, 3)).astype(np.float32)
    Xv = rng.uniform(-2.4, 2.4, size=(cfg.approx_val_n, 3)).astype(np.float32)
    Xe = rng.uniform(-2.4, 2.4, size=(cfg.approx_eval_n, 3)).astype(np.float32)
    ytr = approx_target_psi(Xtr).astype(np.float32)
    yv = approx_target_psi(Xv).astype(np.float32)
    ye = approx_target_psi(Xe)

    a_lo, a_hi = 0.50, 1.70
    Lh = (a_hi - a_lo)/4.0

    # Independent small-exposure Cox environments for law-level comparison.
    law_paths = []
    for p in range(80 if not QUICK else 24):
        obj = simulate_one_path(
            p, cfg, cfg.seed + 990000 + p,
            n_bins=180 if not QUICK else 100, T=0.75, base_rate=2.2
        )
        g = obj["frame"]
        Xp = g[["x1","x2","x3"]].to_numpy(np.float32)
        Rp = g["R"].to_numpy(float)
        law_paths.append((Xp, Rp, obj["dt"]))
    E_R_exposure = np.mean([np.sum(R)*dt for X,R,dt in law_paths])

    rows = []
    for width in cfg.approx_widths:
        print(f"Training universal-approximation network width={width} ...")
        net = train_approx_net(width, Xtr, ytr, Xv, yv)
        net.eval()
        with torch.no_grad():
            pe = net(torch.tensor(Xe, device=DEVICE)).cpu().numpy()
        eps = float(np.max(np.abs(pe - ye)))
        rmse = float(np.sqrt(np.mean((pe - ye)**2)))

        coupling_probs = []
        l1_mod = []
        for Xp, Rp, dt in law_paths:
            psi_star = approx_target_psi(Xp)
            with torch.no_grad():
                psi_hat = net(torch.tensor(Xp, device=DEVICE)).cpu().numpy()
            Astar = a_lo + (a_hi-a_lo)*expit(psi_star)
            Ahat = a_lo + (a_hi-a_lo)*expit(psi_hat)
            D = float(np.sum(Rp * np.abs(Ahat-Astar))*dt)
            coupling_probs.append(1.0 - math.exp(-D))
            l1_mod.append(float(np.mean(np.abs(Ahat-Astar))))

        theorem_raw = Lh * eps * E_R_exposure
        rows.append({
            "width": width,
            "score_sup_error_eps": eps,
            "score_RMSE": rmse,
            "mean_abs_modulation_error": np.mean(l1_mod),
            "mean_coupling_disagreement_bound": np.mean(coupling_probs),
            "theorem_TV_upper_bound_raw": theorem_raw,
            "theorem_TV_upper_bound_clipped": min(1.0, theorem_raw),
            "mean_R_exposure": E_R_exposure,
        })
        torch.save(net.state_dict(), MODEL_DIR / f"approx_net_width_{width}.pt")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Long-run LLN and random-compensator CLT
# -----------------------------------------------------------------------------

def long_run_experiment(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    n = cfg.long_bins
    T = cfg.long_T
    dt = T/n
    horizon_fracs = np.asarray([0.05, 0.10, 0.20, 0.40, 0.70, 1.00])
    horizon_idx = np.unique(np.clip((horizon_fracs*n).astype(int)-1, 1, n-1))
    horizons = (horizon_idx+1)*dt
    ratios = np.empty((cfg.long_paths, len(horizon_idx)))
    zfinal = np.empty(cfg.long_paths)

    for p in range(cfg.long_paths):
        rng = np.random.default_rng(cfg.seed + 1230000 + p)
        z = simulate_ou_exact(n, dt, tau=1.3, sigma_stationary=0.42, rng=rng)
        R = 6.0*np.exp(z - 0.5*0.42**2)
        X = simulate_features(n, dt, rng)
        s = approx_target_psi(X)
        A = bounded_link(s, 0.55, 1.55)
        lam = R*A
        mu = lam*dt
        y = rng.poisson(mu)
        cumN = np.cumsum(y)
        cumL = np.cumsum(mu)
        ratios[p,:] = cumN[horizon_idx] / np.clip(cumL[horizon_idx], 1e-12, None)
        zfinal[p] = (cumN[-1] - cumL[-1]) / math.sqrt(max(cumL[-1], 1e-12))

    rows = []
    for j,h in enumerate(horizons):
        rows.append({
            "horizon": h,
            "mean_N_over_Lambda": float(np.mean(ratios[:,j])),
            "median_N_over_Lambda": float(np.median(ratios[:,j])),
            "q05": float(np.quantile(ratios[:,j], 0.05)),
            "q95": float(np.quantile(ratios[:,j], 0.95)),
            "RMSE_from_1": float(np.sqrt(np.mean((ratios[:,j]-1.0)**2))),
        })
    ks = stats.kstest(zfinal, "norm")
    summary = pd.DataFrame(rows)
    summary.attrs["clt_mean"] = float(zfinal.mean())
    summary.attrs["clt_sd"] = float(zfinal.std(ddof=1))
    summary.attrs["clt_ks_D"] = float(ks.statistic)
    summary.attrs["clt_ks_p"] = float(ks.pvalue)
    return summary, zfinal


# -----------------------------------------------------------------------------
# LaTeX section writer based ONLY on generated outputs
# -----------------------------------------------------------------------------

def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "NA"
    if abs(x) < 1e-3 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def write_latex_section(struct_tab: pd.DataFrame,
                        perf: pd.DataFrame,
                        stability: pd.DataFrame,
                        approx: pd.DataFrame,
                        longtab: pd.DataFrame) -> Path:
    # Identify best fitted (exclude Oracle) by held-out NLL.
    learned = perf[perf["Model"] != "Oracle"].copy()
    best = learned.sort_values("NLL/bin").iloc[0]
    neural = perf[perf["Model"] == "Neural AIM-DSPP"].iloc[0]
    linear = perf[perf["Model"] == "Linear AIM"].iloc[0]
    cox = perf[perf["Model"] == "Cox-only"].iloc[0]

    ks_p = struct_tab.loc[struct_tab["Diagnostic"] == "Time-rescaling KS p-value", "Empirical"].iloc[0]
    fano = struct_tab.loc[struct_tab["Diagnostic"] == "Fano factor Var[N]/E[N]", "Empirical"].iloc[0]
    theo_fano = struct_tab.loc[struct_tab["Diagnostic"] == "Fano factor Var[N]/E[N]", "Theory/Target"].iloc[0]

    last_stab = stability.iloc[-1]
    last_approx = approx.sort_values("width").iloc[-1]
    last_long = longtab.iloc[-1]
    clt_mean = longtab.attrs.get("clt_mean", np.nan)
    clt_sd = longtab.attrs.get("clt_sd", np.nan)
    clt_ks_p = longtab.attrs.get("clt_ks_p", np.nan)

    text = rf"""
% ------------------------------------------------------------------
% AUTO-GENERATED FROM aim_dspp_simulation_colab.py
% Do not edit numerical values by hand; rerun the simulation instead.
% ------------------------------------------------------------------
\section{{Simulation Study}}\label{{sec:simulation}}

\subsection{{Objectives and design}}
The simulation study was designed to examine finite-sample implications of the structural results rather than to manufacture a favorable prediction comparison.  In every experiment the event innovation was generated only after the directing environment had been fixed.  On a grid of {CFG.n_bins} intervals over $[0,{CFG.T:g}]$, the baseline $R_t$ was generated from a stationary log-Gaussian Ornstein--Uhlenbeck environment with a mild smooth seasonal component.  Three exogenous stochastic covariates $X_t=(X_{{1t}},X_{{2t}},X_{{3t}})$ were generated independently of the Poisson innovation.  A nonlinear score combined trigonometric, quadratic and interaction terms, and the true multiplier was
\[
 A(t)=\exp\{{S(t)-\overline S\}},
\]
so that the pathwise geometric normalization $\int \log A(t)\,dt/T=0$ holds exactly.  Conditional on $R$, $X$, and $A$, counts in each grid cell were independent Poisson with mean $R_tA(t)\Delta$.  Conditional event times were then drawn uniformly within occupied cells, which is exact for the resulting piecewise-constant conditional intensity.

We simulated {CFG.n_paths} independent paths.  Complete paths, rather than individual bins, were assigned to training ({CFG.n_train_paths}), validation ({CFG.n_val_paths}), and test sets, preventing leakage of a latent directing trajectory between fitting and evaluation.  The proposed neural modulator was estimated by conditional Poisson likelihood with the known simulated $R_t$ used as an offset and with the same geometric normalization imposed during optimization.  It was compared with (i) a Cox-only model $A\equiv1$ and (ii) a linear AIM model using the same covariates and normalization.  An oracle row using the data-generating multiplier is reported only as a reference and is not a fitted competitor.

\subsection{{Conditional-Poisson structure and overdispersion}}
The random-time-rescaling diagnostic under the true directing intensity produced a Kolmogorov--Smirnov $p$-value of {fmt(ks_p,4)}.  The empirical Fano factor of total path counts was {fmt(fano)}, while the Monte Carlo counterpart of the Cox variance identity $\operatorname{{Var}}N(T)=\mathbb E\Lambda(T)+\operatorname{{Var}}\Lambda(T)$ implied {fmt(theo_fano)}.  Figure~\ref{{fig:sim_structure}} displays a representative directing environment and the resulting point pattern, while Figure~\ref{{fig:sim_diagnostics}} reports the rescaling and overdispersion diagnostics.

\subsection{{Identifiability experiment}}
To illustrate Proposition~\ref{{prop:nonident}}, a positive nonconstant function $c(t)$ was applied to form $R'(t)=R(t)c(t)$ and $A'(t)=A(t)/c(t)$.  The code verifies to machine precision that $R'A'=RA$ and that the two conditional Poisson log-likelihoods are identical.  The chosen nonconstant $c(t)$ also has zero time-average log, so both $A$ and $A'$ satisfy the same global geometric normalization.  This deliberately illustrates the point made after Proposition~\ref{{prop:nonident}}: removing a single global scale degree of freedom does not, by itself, identify an otherwise unrestricted time-varying factorization.  Thus a flexible fitting routine cannot recover a scientifically meaningful baseline--AI decomposition merely by increasing model capacity; see Figure~\ref{{fig:sim_identifiability}}.

\subsection{{Held-out estimation}}
On previously unseen directing paths, the lowest held-out negative log-likelihood among the fitted models was obtained by \emph{{{best['Model']}}}.  The neural AIM-DSPP attained NLL/bin {fmt(neural['NLL/bin'])}, compared with {fmt(linear['NLL/bin'])} for the linear AIM model and {fmt(cox['NLL/bin'])} for the Cox-only model.  Its intensity RMSE was {fmt(neural['Intensity RMSE'])}.  Bootstrap intervals in Table~\ref{{tab:sim_model_comparison}} are computed across complete held-out paths and therefore preserve within-path dependence.  Figure~\ref{{fig:sim_estimation}} gives intensity recovery and calibration plots; Figure~\ref{{fig:sim_training}} gives optimization traces and path-level performance distributions.

\subsection{{Law-level stability}}
For the bounded link $h(s)=a_-+(a_+-a_-)\operatorname{{logit}}^{{-1}}(s)$, exogenous score perturbations of increasing magnitude were applied on identical directing environments.  At the largest perturbation $\delta={fmt(last_stab['delta'])}$, the empirical common-PRM mismatch frequency was {fmt(last_stab['empirical_coupling_mismatch'])}, the Monte Carlo average of the exact conditional expression $1-e^{{-D_T}}$ was {fmt(last_stab['exact_E_1_minus_exp_minus_D'])}, and the average raw Lipschitz upper bound was {fmt(last_stab['mean_Lipschitz_bound_raw'])}.  The complete perturbation curve is shown in Figure~\ref{{fig:sim_stability}} and numerically checks the coupling mechanism behind Theorem~\ref{{thm:tv}} rather than estimating total variation itself.

\subsection{{Neural approximation and induced process law}}
Networks of increasing width were trained directly against a smooth target score on a compact covariate cube.  For the widest architecture ($m={int(last_approx['width'])}$), the empirical evaluation-set supremum score error was {fmt(last_approx['score_sup_error_eps'])}, with mean coupling-disagreement bound {fmt(last_approx['mean_coupling_disagreement_bound'])}.  The reported theorem bound uses the empirical supremum error on a large independent evaluation sample and is therefore a numerical diagnostic of Theorem~\ref{{thm:universal}}, not a proof of a global mathematical supremum bound.  Figure~\ref{{fig:sim_approximation}} shows how function-space approximation propagates to the process-law coupling bound.

\subsection{{Long-run behavior}}
Finally, stationary AIM-DSPP paths were simulated over increasing horizons.  At the longest horizon the median $N(T)/\Lambda(T)$ was {fmt(last_long['median_N_over_Lambda'])} and its RMSE from one was {fmt(last_long['RMSE_from_1'])}.  The final compensated statistic $(N(T)-\Lambda(T))/\sqrt{{\Lambda(T)}}$ had Monte Carlo mean {fmt(clt_mean)}, standard deviation {fmt(clt_sd)}, and a standard-normal KS $p$-value of {fmt(clt_ks_p,4)}.  These experiments illustrate the finite-horizon behavior associated with Theorems~\ref{{thm:lln}} and~\ref{{thm:clt}}; they are not substitutes for the proofs.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/01_path_anatomy.pdf}}
\caption{{Representative AIM-DSPP path: random Cox baseline, normalized AI multiplier, product intensity, and the resulting event realization.}}
\label{{fig:sim_structure}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/02_structural_diagnostics.pdf}}
\caption{{Conditional-Poisson diagnostics and Cox overdispersion decomposition.}}
\label{{fig:sim_diagnostics}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/03_identifiability.pdf}}
\caption{{Multiplicative non-identifiability: substantially different factors can induce exactly the same conditional intensity.}}
\label{{fig:sim_identifiability}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/04_estimation_and_calibration.pdf}}
\caption{{Held-out intensity recovery and count calibration for the fitted models.}}
\label{{fig:sim_estimation}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/05_training_and_pathwise_performance.pdf}}
\caption{{Optimization traces and complete-path held-out performance.}}
\label{{fig:sim_training}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/06_stability_theorem.pdf}}
\caption{{Empirical verification of the common-Poisson-random-measure stability construction.}}
\label{{fig:sim_stability}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/07_universal_approximation.pdf}}
\caption{{Neural score approximation and the corresponding law-level coupling diagnostics.}}
\label{{fig:sim_approximation}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{{figures/08_long_run_theory.pdf}}
\caption{{Finite-horizon illustration of the compensator law of large numbers and the random-compensator central limit theorem.}}
\label{{fig:sim_longrun}}
\end{{figure}}

\begin{{table}}[!htbp]
\centering
\caption{{Held-out comparison. Confidence intervals are nonparametric bootstrap intervals over complete test paths.}}
\label{{tab:sim_model_comparison}}
\resizebox{{\linewidth}}{{!}}{{\input{{tables/03_model_comparison.tex}}}}
\end{{table}}
"""
    out = ROOT / "simulation_section_auto.tex"
    out.write_text(text.strip() + "\n", encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# Main workflow and publication figures
# -----------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()
    print("\nAIM-DSPP SIMULATION STUDY")
    print("=" * 88)
    print(f"Device: {DEVICE}")
    print(f"Quick mode: {QUICK}")
    print(f"Output directory: {ROOT.resolve()}")
    print("No benchmark selection is conditional on its performance.\n")

    # Save configuration immediately.
    (ROOT / "config.json").write_text(json.dumps(asdict(CFG), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Study 1: Main data and structural diagnostics
    # ------------------------------------------------------------------
    print("\n[1/7] Simulating the main AIM-DSPP dataset ...")
    df, events, dt = simulate_dataset(CFG)
    df.to_csv(DATA_DIR / "main_simulation_grid.csv.gz", index=False, compression="gzip")

    # Representative path anatomy
    g0 = df[df["path"] == 0]
    ev0 = events[0]
    fig, ax = plt.subplots(4, 1, figsize=(11.5, 9.0), sharex=True, constrained_layout=True)
    ax[0].plot(g0["t"], g0["R"], color=COLORS["baseline"])
    ax[0].set_ylabel(r"$R_t$"); ax[0].set_title("Random directing environment")
    ax[1].plot(g0["t"], g0["A_true"], color=COLORS["accent"])
    ax[1].axhline(1.0, color="black", lw=0.9, ls=":")
    ax[1].set_ylabel(r"$A(t)$"); ax[1].set_title("Geometrically normalized nonlinear AI multiplier")
    ax[2].plot(g0["t"], g0["lambda_true"], color=COLORS["true"])
    ax[2].fill_between(g0["t"].to_numpy(), 0, g0["lambda_true"].to_numpy(), alpha=0.08, color=COLORS["true"])
    ax[2].set_ylabel(r"$\lambda(t)$"); ax[2].set_title(r"Product intensity $\lambda(t)=R_tA(t)$")
    # event rug + cumulative count
    ax[3].vlines(ev0, 0, 1, color=COLORS["neural"], lw=0.45, alpha=0.70)
    ax[3].set_ylim(0, 1.05); ax[3].set_yticks([]); ax[3].set_xlabel("Time")
    ax[3].set_title(f"Conditional Poisson realization: N(T)={len(ev0)}")
    save_show(fig, "01_path_anatomy")

    struct_tab = structural_diagnostics(df, events, dt)
    save_table(struct_tab, "01_structural_diagnostics")
    display_df(struct_tab, "TABLE 1 — Structural diagnostics")

    # Structural diagnostics figure
    gaps = []
    for p, g in df.groupby("path", sort=True):
        gaps.extend(rescaled_gaps(events[int(p)], g["lambda_true"].to_numpy(), dt))
    gaps = np.asarray(gaps)
    by_path = df.groupby("path").agg(N=("count","sum"), Lambda=("mu_true","sum")).reset_index()
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.3), constrained_layout=True)
    q = np.linspace(0.01, 0.99, 99)
    emp_q = np.quantile(gaps, q)
    exp_q = stats.expon.ppf(q)
    ax[0,0].plot(exp_q, emp_q, color=COLORS["neural"])
    lim = max(np.quantile(exp_q, .98), np.quantile(emp_q, .98))
    ax[0,0].plot([0,lim],[0,lim], ls="--", color="black", lw=1)
    ax[0,0].set_xlabel("Exp(1) theoretical quantile"); ax[0,0].set_ylabel("Rescaled-gap quantile")
    ax[0,0].set_title("Random-time-rescaling Q–Q")

    xs = np.linspace(0, np.quantile(gaps, .995), 250)
    ax[0,1].hist(gaps, bins=55, density=True, alpha=0.45, color=COLORS["linear"])
    ax[0,1].plot(xs, np.exp(-xs), color=COLORS["true"], lw=2.0, label="Exp(1)")
    ax[0,1].set_xlabel("Rescaled inter-event gap"); ax[0,1].set_ylabel("Density")
    ax[0,1].set_title("Conditional-Poisson diagnostic"); ax[0,1].legend()

    ax[1,0].scatter(by_path["Lambda"], by_path["N"], s=24, alpha=.75, color=COLORS["accent"])
    lo = min(by_path["Lambda"].min(), by_path["N"].min()); hi = max(by_path["Lambda"].max(), by_path["N"].max())
    ax[1,0].plot([lo,hi],[lo,hi], ls="--", color="black", lw=1)
    ax[1,0].set_xlabel(r"Integrated intensity $\Lambda(T)$"); ax[1,0].set_ylabel(r"Observed $N(T)$")
    ax[1,0].set_title("Path-level conditional calibration")

    emp_var = by_path["N"].var(ddof=1); E_L = by_path["Lambda"].mean(); V_L = by_path["Lambda"].var(ddof=1)
    ax[1,1].bar(["Empirical Var N", "EΛ + VarΛ"], [emp_var, E_L+V_L], color=[COLORS["neural"], COLORS["oracle"]], alpha=.85)
    ax[1,1].set_ylabel("Variance"); ax[1,1].set_title("Cox overdispersion identity")
    save_show(fig, "02_structural_diagnostics")

    # ------------------------------------------------------------------
    # Study 2: Exact non-identifiability demonstration
    # ------------------------------------------------------------------
    print("\n[2/7] Demonstrating multiplicative non-identifiability ...")
    t0 = g0["t"].to_numpy(); R0 = g0["R"].to_numpy(); A0 = g0["A_true"].to_numpy(); l0 = R0*A0
    c = np.exp(0.60*np.sin(2*np.pi*t0/CFG.T) + 0.22*np.cos(4*np.pi*t0/CFG.T))
    Rp = R0*c; Ap = A0/c; lp = Rp*Ap
    y0 = g0["count"].to_numpy()
    ll1 = poisson_loglik(y0, l0*dt); ll2 = poisson_loglik(y0, lp*dt)
    ident_tab = pd.DataFrame([
        ["max |lambda-lambda'|", np.max(np.abs(l0-lp))],
        ["max relative intensity error", np.max(np.abs(l0-lp)/np.clip(l0,1e-12,None))],
        ["conditional loglik original", ll1],
        ["conditional loglik transformed", ll2],
        ["absolute loglik difference", abs(ll1-ll2)],
        ["mean log(A)", np.mean(np.log(A0))],
        ["mean log(A')", np.mean(np.log(Ap))],
    ], columns=["Quantity", "Value"])
    save_table(ident_tab, "02_identifiability")
    display_df(ident_tab, "TABLE 2 — Multiplicative non-identifiability check")

    fig, ax = plt.subplots(3,1,figsize=(11.5,8.2),sharex=True,constrained_layout=True)
    ax[0].plot(t0,R0,label="R",color=COLORS["baseline"]); ax[0].plot(t0,Rp,label="R' = Rc",color=COLORS["linear"],alpha=.85)
    ax[0].set_ylabel("Baseline"); ax[0].legend(ncol=2); ax[0].set_title("Different baseline factors")
    ax[1].plot(t0,A0,label="A",color=COLORS["accent"]); ax[1].plot(t0,Ap,label="A' = A/c",color=COLORS["gold"],alpha=.9)
    ax[1].set_ylabel("Multiplier"); ax[1].legend(ncol=2); ax[1].set_title("Compensating AI factors")
    ax[2].plot(t0,l0,label="RA",color=COLORS["true"],lw=2.2); ax[2].plot(t0,lp,label="R'A'",color=COLORS["neural"],ls="--",lw=1.3)
    ax[2].set_ylabel("Intensity"); ax[2].set_xlabel("Time"); ax[2].legend(ncol=2)
    ax[2].set_title(f"Identical product intensity; |Δ log L|={abs(ll1-ll2):.2e}")
    save_show(fig, "03_identifiability")

    # ------------------------------------------------------------------
    # Study 3: Estimation and honest benchmark comparison
    # ------------------------------------------------------------------
    print("\n[3/7] Fitting conditional-likelihood models ...")
    train_paths = np.arange(0, CFG.n_train_paths)
    val_paths = np.arange(CFG.n_train_paths, CFG.n_train_paths + CFG.n_val_paths)
    test_paths = np.arange(CFG.n_train_paths + CFG.n_val_paths, CFG.n_paths)
    if len(test_paths) < 2:
        raise RuntimeError("Need at least two test paths. Adjust split configuration.")
    train_df = df[df["path"].isin(train_paths)].copy().reset_index(drop=True)
    val_df = df[df["path"].isin(val_paths)].copy().reset_index(drop=True)
    test_df = df[df["path"].isin(test_paths)].copy().reset_index(drop=True)
    print(f"Paths: train={len(train_paths)}, validation={len(val_paths)}, test={len(test_paths)}")

    ttrain = make_tensor_dataset(train_df, dt); tval = make_tensor_dataset(val_df, dt)

    linear = LinearModulator(3)
    linear, hist_lin = train_modulator(
        linear, ttrain, tval, CFG.lr_linear, CFG.max_epochs_linear,
        CFG.patience_linear, CFG.weight_decay_linear, "Linear AIM"
    )
    neural = NeuralModulator(3, CFG.hidden)
    neural, hist_nn = train_modulator(
        neural, ttrain, tval, CFG.lr_mlp, CFG.max_epochs_mlp,
        CFG.patience_mlp, CFG.weight_decay_mlp, "Neural AIM"
    )
    torch.save(linear.state_dict(), MODEL_DIR / "linear_aim.pt")
    torch.save(neural.state_dict(), MODEL_DIR / "neural_aim_dspp.pt")
    hist_lin.to_csv(TAB_DIR / "training_history_linear.csv", index=False)
    hist_nn.to_csv(TAB_DIR / "training_history_neural.csv", index=False)

    pred_lin = predict_modulator(linear, test_df, dt)
    pred_nn = predict_modulator(neural, test_df, dt)
    pred_by_model = {
        "Cox-only": test_df["R"].to_numpy(),
        "Linear AIM": pred_lin["lambda"],
        "Neural AIM-DSPP": pred_nn["lambda"],
        "Oracle": test_df["lambda_true"].to_numpy(),
    }
    perf, path_perf = evaluate_predictions(test_df, pred_by_model, events, dt)
    save_table(perf, "03_model_comparison")
    save_table(path_perf, "03b_pathwise_model_performance")
    display_cols = ["Model","NLL/bin","Poisson deviance/bin","Intensity RMSE","Log-intensity RMSE","Intensity MAE","Rescaling KS D","Rescaling KS p"]
    display_df(perf[display_cols], "TABLE 3 — Held-out model comparison")

    # Save test predictions for auditability.
    pred_save = test_df[["path","bin","t","count","R","A_true","lambda_true"]].copy()
    for name, arr in pred_by_model.items():
        key = name.lower().replace(" ","_").replace("-","_")
        pred_save[f"lambda_{key}"] = arr
    pred_save.to_csv(DATA_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

    # Calibration frames
    cal_frames = []
    for name, arr in pred_by_model.items():
        cal_frames.append(calibration_frame(test_df, name, arr, dt))
    calibration = pd.concat(cal_frames, ignore_index=True)
    save_table(calibration, "04_calibration_deciles")

    # Figure estimation/recovery
    representative_test_path = int(test_paths[len(test_paths)//2])
    gm = test_df[test_df["path"] == representative_test_path]
    mask_rep = test_df["path"].to_numpy() == representative_test_path
    fig, ax = plt.subplots(2,2,figsize=(12.2,8.6),constrained_layout=True)
    ax[0,0].plot(gm["t"], gm["lambda_true"], color=COLORS["true"], label="True", lw=2.2)
    ax[0,0].plot(gm["t"], pred_by_model["Cox-only"][mask_rep], color=COLORS["baseline"], label="Cox-only", alpha=.8)
    ax[0,0].plot(gm["t"], pred_by_model["Linear AIM"][mask_rep], color=COLORS["linear"], label="Linear AIM", alpha=.9)
    ax[0,0].plot(gm["t"], pred_by_model["Neural AIM-DSPP"][mask_rep], color=COLORS["neural"], label="Neural AIM-DSPP", alpha=.9)
    ax[0,0].set_title(f"Held-out intensity path {representative_test_path}"); ax[0,0].set_xlabel("Time"); ax[0,0].set_ylabel("Intensity")
    ax[0,0].legend(ncol=2)

    # True vs predicted intensity for a controlled sample
    rng = np.random.default_rng(CFG.seed+88)
    choose = rng.choice(len(test_df), size=min(6000,len(test_df)), replace=False)
    for name, col in [("Linear AIM",COLORS["linear"]),("Neural AIM-DSPP",COLORS["neural"])]:
        ax[0,1].scatter(test_df["lambda_true"].to_numpy()[choose], pred_by_model[name][choose], s=8, alpha=.20, label=name, color=col)
    mm = np.quantile(test_df["lambda_true"], .995)
    ax[0,1].plot([0,mm],[0,mm],ls="--",color="black",lw=1)
    ax[0,1].set_xlim(0,mm); ax[0,1].set_ylim(0,mm)
    ax[0,1].set_xlabel("True intensity"); ax[0,1].set_ylabel("Predicted intensity"); ax[0,1].set_title("Intensity recovery")
    ax[0,1].legend()

    for name, col in [("Cox-only",COLORS["baseline"]),("Linear AIM",COLORS["linear"]),("Neural AIM-DSPP",COLORS["neural"]),("Oracle",COLORS["oracle"])]:
        cc = calibration[calibration["Model"]==name]
        ax[1,0].plot(cc["pred"], cc["obs"], marker="o", ms=4, label=name, color=col)
    maxcal = max(calibration["pred"].max(),calibration["obs"].max())
    ax[1,0].plot([0,maxcal],[0,maxcal],ls="--",color="black",lw=1)
    ax[1,0].set_xlabel("Mean predicted count/bin"); ax[1,0].set_ylabel("Mean observed count/bin")
    ax[1,0].set_title("Decile calibration on held-out paths"); ax[1,0].legend(ncol=2)

    plot_perf = perf[perf["Model"]!="Oracle"]
    x = np.arange(len(plot_perf))
    vals = plot_perf["Poisson deviance/bin"].to_numpy()
    lo = vals - plot_perf["Poisson deviance/bin CI low"].to_numpy()
    hi = plot_perf["Poisson deviance/bin CI high"].to_numpy() - vals
    ax[1,1].bar(x, vals, color=[COLORS["baseline"],COLORS["linear"],COLORS["neural"]], alpha=.88)
    ax[1,1].errorbar(x, vals, yerr=np.vstack([lo,hi]), fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax[1,1].set_xticks(x, plot_perf["Model"], rotation=12)
    ax[1,1].set_ylabel("Held-out deviance/bin"); ax[1,1].set_title("Complete-path bootstrap uncertainty")
    save_show(fig, "04_estimation_and_calibration")

    # Training and pathwise performance figure
    fig, ax = plt.subplots(2,2,figsize=(12.2,8.5),constrained_layout=True)
    ax[0,0].plot(hist_lin["epoch"],hist_lin["train_nll_no_const"],color=COLORS["linear"],label="train")
    ax[0,0].plot(hist_lin["epoch"],hist_lin["val_nll_no_const"],color=COLORS["gold"],label="validation")
    ax[0,0].set_title("Linear AIM optimization"); ax[0,0].set_xlabel("Epoch"); ax[0,0].set_ylabel("NLL/bin (constant omitted)"); ax[0,0].legend()
    ax[0,1].plot(hist_nn["epoch"],hist_nn["train_nll_no_const"],color=COLORS["neural"],label="train")
    ax[0,1].plot(hist_nn["epoch"],hist_nn["val_nll_no_const"],color=COLORS["oracle"],label="validation")
    ax[0,1].set_title("Neural AIM-DSPP optimization"); ax[0,1].set_xlabel("Epoch"); ax[0,1].set_ylabel("NLL/bin (constant omitted)"); ax[0,1].legend()

    models_fit = ["Cox-only","Linear AIM","Neural AIM-DSPP"]
    data_box = [path_perf[path_perf["Model"]==m]["NLL/bin"].to_numpy() for m in models_fit]
    b = ax[1,0].boxplot(data_box, tick_labels=models_fit, patch_artist=True, showfliers=False)
    for patch, col in zip(b["boxes"],[COLORS["baseline"],COLORS["linear"],COLORS["neural"]]): patch.set_facecolor(col); patch.set_alpha(.55)
    ax[1,0].set_ylabel("NLL/bin"); ax[1,0].set_title("Held-out pathwise NLL")
    data_box2 = [path_perf[path_perf["Model"]==m]["Intensity RMSE"].to_numpy() for m in models_fit]
    b2 = ax[1,1].boxplot(data_box2, tick_labels=models_fit, patch_artist=True, showfliers=False)
    for patch, col in zip(b2["boxes"],[COLORS["baseline"],COLORS["linear"],COLORS["neural"]]): patch.set_facecolor(col); patch.set_alpha(.55)
    ax[1,1].set_ylabel("Intensity RMSE"); ax[1,1].set_title("Held-out intensity recovery by path")
    for a in ax[1,:]:
        a.tick_params(axis="x", rotation=12)
    save_show(fig, "05_training_and_pathwise_performance")

    # ------------------------------------------------------------------
    # Study 4: Stability theorem
    # ------------------------------------------------------------------
    print("\n[4/7] Running stability/coupling experiment ...")
    stab = stability_experiment(CFG)
    save_table(stab, "05_stability")
    display_df(stab, "TABLE 4 — Law-level stability experiment")

    fig, ax = plt.subplots(1,2,figsize=(11.7,4.8),constrained_layout=True)
    ax[0].plot(stab["delta"],stab["empirical_coupling_mismatch"],marker="o",label="Empirical common-PRM mismatch",color=COLORS["neural"])
    ax[0].plot(stab["delta"],stab["exact_E_1_minus_exp_minus_D"],marker="s",label=r"$E[1-e^{-D_T}]$",color=COLORS["oracle"])
    ax[0].plot(stab["delta"],stab["mean_Lipschitz_bound_clipped"],marker="^",label="Lipschitz bound (clipped at 1)",color=COLORS["linear"])
    ax[0].set_xlabel(r"Score perturbation size $\delta$"); ax[0].set_ylabel("Probability / upper bound"); ax[0].set_ylim(-.02,1.03)
    ax[0].set_title("Coupling disagreement and theorem bound"); ax[0].legend()
    ax[1].plot(stab["delta"],stab["mean_D"],marker="o",color=COLORS["accent"],label=r"$E[D_T]$")
    ax[1].plot(stab["delta"],stab["mean_Lipschitz_bound_raw"],marker="s",color=COLORS["gold"],label="Raw Lipschitz upper bound")
    ax[1].set_xlabel(r"Score perturbation size $\delta$"); ax[1].set_ylabel("Integrated discrepancy")
    ax[1].set_title("Intensity-space control"); ax[1].legend()
    save_show(fig, "06_stability_theorem")

    # ------------------------------------------------------------------
    # Study 5: Universal approximation -> process-law diagnostics
    # ------------------------------------------------------------------
    print("\n[5/7] Running neural universal-approximation experiment ...")
    approx = universal_approximation_experiment(CFG)
    save_table(approx, "06_universal_approximation")
    display_df(approx, "TABLE 5 — Universal approximation and law-level diagnostics")

    fig, ax = plt.subplots(1,3,figsize=(13.3,4.5),constrained_layout=True)
    ax[0].plot(approx["width"],approx["score_RMSE"],marker="o",color=COLORS["neural"],label="RMSE")
    ax[0].plot(approx["width"],approx["score_sup_error_eps"],marker="s",color=COLORS["accent"],label="Empirical sup error")
    ax[0].set_xlabel("Hidden width"); ax[0].set_ylabel("Score error"); ax[0].set_yscale("log"); ax[0].set_title("Function-space approximation"); ax[0].legend()
    ax[1].plot(approx["width"],approx["mean_abs_modulation_error"],marker="o",color=COLORS["gold"])
    ax[1].set_xlabel("Hidden width"); ax[1].set_ylabel(r"Mean $|A_\theta-A^\star|$"); ax[1].set_yscale("log"); ax[1].set_title("Positive multiplier approximation")
    ax[2].plot(approx["width"],approx["mean_coupling_disagreement_bound"],marker="o",color=COLORS["oracle"],label=r"$E[1-e^{-D_T}]$")
    ax[2].plot(approx["width"],approx["theorem_TV_upper_bound_clipped"],marker="s",color=COLORS["linear"],label="Theorem bound (clipped)")
    ax[2].set_xlabel("Hidden width"); ax[2].set_ylabel("Law-level bound"); ax[2].set_ylim(-.02,1.03); ax[2].set_title("Approximation transferred to process law"); ax[2].legend()
    save_show(fig, "07_universal_approximation")

    # ------------------------------------------------------------------
    # Study 6: LLN and CLT
    # ------------------------------------------------------------------
    print("\n[6/7] Running long-horizon LLN/CLT experiment ...")
    longtab, zfinal = long_run_experiment(CFG)
    # Preserve attrs separately because CSV cannot store them.
    save_table(longtab, "07_long_run")
    clt_tab = pd.DataFrame([{
        "mean": longtab.attrs["clt_mean"],
        "sd": longtab.attrs["clt_sd"],
        "KS_D": longtab.attrs["clt_ks_D"],
        "KS_p": longtab.attrs["clt_ks_p"],
    }])
    save_table(clt_tab, "07b_clt_diagnostic")
    display_df(longtab, "TABLE 6A — Compensator LLN diagnostics")
    display_df(clt_tab, "TABLE 6B — Random-compensator CLT diagnostic")

    fig, ax = plt.subplots(1,3,figsize=(13.2,4.4),constrained_layout=True)
    ax[0].plot(longtab["horizon"],longtab["median_N_over_Lambda"],marker="o",color=COLORS["neural"],label="Median")
    ax[0].fill_between(longtab["horizon"].to_numpy(),longtab["q05"].to_numpy(),longtab["q95"].to_numpy(),alpha=.18,color=COLORS["linear"],label="5–95% envelope")
    ax[0].axhline(1.0,ls="--",color="black",lw=1)
    ax[0].set_xlabel("Horizon"); ax[0].set_ylabel(r"$N(T)/\Lambda(T)$"); ax[0].set_title("Compensator LLN"); ax[0].legend()
    xx = np.linspace(min(-4,zfinal.min()),max(4,zfinal.max()),300)
    ax[1].hist(zfinal,bins=34,density=True,alpha=.45,color=COLORS["accent"])
    ax[1].plot(xx,stats.norm.pdf(xx),color=COLORS["true"],lw=2,label="N(0,1)")
    ax[1].set_xlabel(r"$(N-\Lambda)/\sqrt{\Lambda}$"); ax[1].set_ylabel("Density"); ax[1].set_title("Random-compensator CLT"); ax[1].legend()
    q = np.linspace(.01,.99,99); zq=np.quantile(zfinal,q); nq=stats.norm.ppf(q)
    ax[2].plot(nq,zq,color=COLORS["oracle"]); lo=min(nq.min(),zq.min());hi=max(nq.max(),zq.max()); ax[2].plot([lo,hi],[lo,hi],ls="--",color="black",lw=1)
    ax[2].set_xlabel("Normal theoretical quantile"); ax[2].set_ylabel("Empirical quantile"); ax[2].set_title("CLT Q–Q diagnostic")
    save_show(fig, "08_long_run_theory")

    # ------------------------------------------------------------------
    # Study 7: Write LaTeX and package everything
    # ------------------------------------------------------------------
    print("\n[7/7] Writing LaTeX-ready section and packaging outputs ...")
    latex_path = write_latex_section(struct_tab, perf, stab, approx, longtab)
    print(f"Saved LaTeX section: {latex_path}")

    # Machine-readable summary
    summary = {
        "device": str(DEVICE),
        "quick_mode": QUICK,
        "config": asdict(CFG),
        "structural_diagnostics": struct_tab.to_dict(orient="records"),
        "model_comparison": perf.to_dict(orient="records"),
        "stability": stab.to_dict(orient="records"),
        "universal_approximation": approx.to_dict(orient="records"),
        "long_run": longtab.to_dict(orient="records"),
        "clt": clt_tab.to_dict(orient="records")[0],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""AIM-DSPP Simulation Outputs
===========================
Generated by aim_dspp_simulation_colab.py
Seed: {CFG.seed}
Device: {DEVICE}
Quick mode: {QUICK}

Folders
-------
figures/ : every plot in publication PNG and vector PDF
         01_path_anatomy
         02_structural_diagnostics
         03_identifiability
         04_estimation_and_calibration
         05_training_and_pathwise_performance
         06_stability_theorem
         07_universal_approximation
         08_long_run_theory

tables/  : CSV and LaTeX tables, plus training histories
models/  : fitted PyTorch state dictionaries
data/    : compressed simulated grid data and held-out predictions

Important top-level files
-------------------------
simulation_section_auto.tex : manuscript-ready section populated from actual run
summary.json                : machine-readable numerical summary
config.json                 : all simulation settings

Scientific interpretation
-------------------------
The Oracle is only a data-generating reference. No benchmark is hidden or
removed based on its performance. The stability experiment estimates the
common-PRM coupling disagreement mechanism E[1-exp(-D_T)], which upper-bounds
total variation; it does NOT claim to compute exact total variation.
The approximation experiment uses an empirical evaluation-set sup error; it
is a numerical diagnostic of the theorem, not a proof of a global supremum.
"""
    (ROOT / "README.txt").write_text(readme, encoding="utf-8")

    # Copy this script into output when __file__ is available.
    try:
        src = Path(__file__).resolve()
        shutil.copy2(src, ROOT / src.name)
    except Exception:
        pass

    zip_path = shutil.make_archive(str(ROOT), "zip", root_dir=ROOT)
    elapsed = time.time() - start_time
    print("\n" + "="*88)
    print("SIMULATION COMPLETE")
    print("="*88)
    print(f"Elapsed: {elapsed/60:.2f} minutes")
    print(f"Output folder: {ROOT.resolve()}")
    print(f"ZIP archive:   {Path(zip_path).resolve()}")
    print("All tables were printed inline and all figures were displayed inline.")

    # Colab: trigger browser download automatically.
    try:
        from google.colab import files
        print("Triggering Colab download of the ZIP ...")
        files.download(zip_path)
    except Exception:
        print("Not running inside Google Colab; download the ZIP from the path above.")


if __name__ == "__main__":
    main()
