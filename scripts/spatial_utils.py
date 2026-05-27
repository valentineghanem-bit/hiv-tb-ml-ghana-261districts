"""
spatial_utils.py — Reusable spatial analysis utilities
Nutrition Anaemia Growth Determinants Ghana 261 Districts
Author: Valentine Golden Ghanem | ORCID: 0009-0002-8332-0220

Shared functions used across the pipeline:
  - Spatial weight matrix construction (queen + KNN)
  - Global Moran's I
  - Local Moran's I (LISA) with BH-FDR correction
  - Getis-Ord Gi* hotspot detection
  - Bivariate Moran's I
  - Spatial leave-one-region-out cross-validation splitter
  - District exceedance probability (posterior > threshold)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm


# ── Spatial weight matrix ─────────────────────────────────────────────────────

def build_queen_weights(adjacency_csv: str) -> np.ndarray:
    """
    Load a pre-computed queen-contiguity adjacency list and return a
    row-standardised weight matrix W (n × n, numpy float32).

    Parameters
    ----------
    adjacency_csv : path to CSV with columns [district_i, district_j]
    """
    adj = pd.read_csv(adjacency_csv)
    n = max(adj["district_i"].max(), adj["district_j"].max()) + 1
    W = np.zeros((n, n), dtype=np.float32)
    for _, row in adj.iterrows():
        i, j = int(row["district_i"]), int(row["district_j"])
        W[i, j] = 1.0
        W[j, i] = 1.0
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0     # guard islands
    return W / row_sums


def build_knn_weights(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Build a row-standardised KNN weight matrix from coordinate array.

    Parameters
    ----------
    coords : (n, 2) array of (lon, lat) or (x, y)
    k      : number of nearest neighbours
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                            metric="haversine").fit(np.radians(coords))
    _, indices = nbrs.kneighbors(np.radians(coords))
    W = np.zeros((n, n), dtype=np.float32)
    for i, neighbours in enumerate(indices):
        for j in neighbours[1:]:         # skip self
            W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


# ── Global Moran's I ──────────────────────────────────────────────────────────

def global_moran(x: np.ndarray, W: np.ndarray,
                 n_permutations: int = 999,
                 seed: int = 42) -> dict:
    """
    Compute Global Moran's I with permutation-based p-value.

    Returns
    -------
    dict with keys: I, E_I, Var_I, z_score, p_value_sim
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    x_c = x[mask] - x[mask].mean()
    W_s = W[np.ix_(mask, mask)]

    n = mask.sum()
    S0 = W_s.sum()

    I = (n / S0) * (x_c @ W_s @ x_c) / (x_c @ x_c)
    E_I = -1.0 / (n - 1)

    # Permutation distribution
    perm_I = np.empty(n_permutations)
    for k in range(n_permutations):
        xp = rng.permutation(x_c)
        perm_I[k] = (n / S0) * (xp @ W_s @ xp) / (xp @ xp)

    p_sim = ((perm_I >= I).sum() + 1) / (n_permutations + 1)

    return {
        "I": float(I),
        "E_I": float(E_I),
        "Var_I": float(perm_I.var()),
        "z_score": float((I - E_I) / (perm_I.std() + 1e-12)),
        "p_value_sim": float(p_sim),
    }


# ── Local Moran's I (LISA) ────────────────────────────────────────────────────

def local_moran(x: np.ndarray, W: np.ndarray,
                n_permutations: int = 999,
                fdr_alpha: float = 0.05,
                seed: int = 42) -> pd.DataFrame:
    """
    Local Moran's I with conditional permutation p-values and
    Benjamini-Hochberg FDR correction.

    Returns DataFrame with columns:
        district_idx, Ii, E_Ii, z_i, p_sim, p_bh, quadrant
        quadrant ∈ {HH, LL, HL, LH, NS}
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    Wz = W @ z

    Ii = z * Wz
    m2 = (z ** 2).mean()

    # Conditional permutation p-values
    p_sim = np.ones(n)
    for i in range(n):
        neighbours = np.where(W[i] > 0)[0]
        if len(neighbours) == 0:
            continue
        perm_vals = np.empty(n_permutations)
        others = np.delete(z, i)
        for k in range(n_permutations):
            perm = rng.permutation(others)
            perm_vals[k] = z[i] * (W[i, neighbours] @ perm[neighbours - (neighbours > i)])
        perm_vals[k] = z[i] * np.dot(W[i, neighbours],
                                      rng.permutation(others)[:len(neighbours)])
        p_sim[i] = ((np.abs(perm_vals) >= np.abs(Ii[i])).sum() + 1) / (n_permutations + 1)

    # Benjamini-Hochberg FDR correction
    order = np.argsort(p_sim)
    p_bh  = np.ones(n)
    for rank, idx in enumerate(order, 1):
        p_bh[idx] = min(1.0, p_sim[idx] * n / rank)
    # Enforce monotonicity
    for k in range(n - 2, -1, -1):
        p_bh[order[k]] = min(p_bh[order[k]], p_bh[order[k + 1]])

    # Quadrant classification
    lag_z = Wz
    quadrant = np.where(
        p_bh >= fdr_alpha, "NS",
        np.where(z > 0,
                 np.where(lag_z > 0, "HH", "HL"),
                 np.where(lag_z > 0, "LH", "LL"))
    )

    return pd.DataFrame({
        "district_idx": np.arange(n),
        "Ii": Ii.round(4),
        "E_Ii": float(-1.0 / (n - 1)),
        "z_i": ((Ii - (-1.0 / (n - 1))) / (np.std(Ii) + 1e-12)).round(3),
        "p_sim": p_sim.round(4),
        "p_bh": p_bh.round(4),
        "quadrant": quadrant,
    })


# ── Bivariate Moran's I ───────────────────────────────────────────────────────

def bivariate_moran(x: np.ndarray, y: np.ndarray, W: np.ndarray,
                    n_permutations: int = 999, seed: int = 42) -> dict:
    """
    Bivariate Global Moran's I: I_xy = z_x' W z_y / n.
    Tests whether the spatial lag of y co-clusters with x.
    """
    rng = np.random.default_rng(seed)
    zx = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    zy = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
    n  = len(zx)

    I_xy = float(zx @ (W @ zy) / n)

    perm_I = np.empty(n_permutations)
    for k in range(n_permutations):
        zy_p = rng.permutation(zy)
        perm_I[k] = zx @ (W @ zy_p) / n

    p_sim = ((np.abs(perm_I) >= np.abs(I_xy)).sum() + 1) / (n_permutations + 1)

    return {
        "I_bivariate": round(I_xy, 4),
        "p_value_sim": round(float(p_sim), 4),
        "mean_perm":   round(float(perm_I.mean()), 4),
        "sd_perm":     round(float(perm_I.std()), 4),
    }


# ── Getis-Ord Gi* ─────────────────────────────────────────────────────────────

def getis_ord_gi_star(x: np.ndarray, W: np.ndarray) -> pd.DataFrame:
    """
    Getis-Ord Gi* statistic (includes self in spatial lag, uses full W).
    Returns DataFrame with z_gi_star, p_value, hotspot_class.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_bar = np.nanmean(x)
    S = np.nanstd(x, ddof=1)

    Wi_sum = W.sum(axis=1)          # sum of weights for each i
    Wi2_sum = (W ** 2).sum(axis=1)  # sum of squared weights
    Wx = W @ x                      # spatial lag

    numerator   = Wx - x_bar * Wi_sum
    denominator = S * np.sqrt(
        (n * Wi2_sum - Wi_sum ** 2) / (n - 1) + 1e-12
    )
    z = numerator / denominator
    p = 2 * (1 - norm.cdf(np.abs(z)))

    hotspot_class = np.where(
        z > 2.576,  "Hotspot (99%)",
        np.where(z > 1.960, "Hotspot (95%)",
        np.where(z > 1.645, "Hotspot (90%)",
        np.where(z < -2.576, "Coldspot (99%)",
        np.where(z < -1.960, "Coldspot (95%)",
        np.where(z < -1.645, "Coldspot (90%)",
                 "Not significant")))))
    )

    return pd.DataFrame({
        "district_idx": np.arange(n),
        "z_gi_star": z.round(4),
        "p_value":   p.round(4),
        "hotspot_class": hotspot_class,
    })


# ── Exceedance probability ────────────────────────────────────────────────────

def exceedance_probability(posterior_draws: np.ndarray,
                           threshold: float) -> np.ndarray:
    """
    P(theta_i > threshold) estimated from posterior draws.

    Parameters
    ----------
    posterior_draws : (n_districts, n_draws) array
    threshold       : scalar threshold (e.g. population-weighted national mean)

    Returns
    -------
    (n_districts,) array of exceedance probabilities in [0, 1]
    """
    return (posterior_draws > threshold).mean(axis=1)


def confirmed_hotspots(exc_probs: dict[str, np.ndarray],
                       threshold: float = 0.95) -> np.ndarray:
    """
    Return boolean mask of districts that are confirmed hotspots
    (P > threshold) for ALL outcomes simultaneously.

    Parameters
    ----------
    exc_probs : {outcome_name: (n_districts,) array of exceedance probs}
    threshold : exceedance probability cutoff (default 0.95)
    """
    masks = np.stack([v > threshold for v in exc_probs.values()], axis=0)
    return masks.all(axis=0)


# ── Spatial LOROCV splitter ───────────────────────────────────────────────────

def spatial_lorocv_splits(df: pd.DataFrame,
                           region_col: str = "region") -> list[tuple]:
    """
    Spatial Leave-One-Region-Out Cross-Validation splits.
    Yields (train_idx, test_idx) for each region fold.

    No district contributes to both train and test in any fold.
    """
    regions = df[region_col].unique()
    indices = np.arange(len(df))
    splits  = []
    for region in regions:
        test_mask  = df[region_col] == region
        train_idx  = indices[~test_mask]
        test_idx   = indices[test_mask]
        splits.append((train_idx, test_idx))
    return splits


# ── BYM2 posterior summary ────────────────────────────────────────────────────

def summarise_bym2_posterior(draws: np.ndarray,
                              credible_interval: float = 0.95) -> pd.DataFrame:
    """
    Summarise BYM2 posterior draws per district.

    Parameters
    ----------
    draws : (n_districts, n_draws)
    credible_interval : width of credible interval (default 0.95)

    Returns
    -------
    DataFrame with posterior_mean, posterior_sd, lower_ci, upper_ci, cv
    """
    alpha = (1 - credible_interval) / 2
    lo    = np.quantile(draws, alpha,       axis=1)
    hi    = np.quantile(draws, 1 - alpha,   axis=1)
    mn    = draws.mean(axis=1)
    sd    = draws.std(axis=1)

    return pd.DataFrame({
        "posterior_mean": mn.round(4),
        "posterior_sd":   sd.round(4),
        f"lower_{int(credible_interval*100)}ci": lo.round(4),
        f"upper_{int(credible_interval*100)}ci": hi.round(4),
        "cv":             (sd / (np.abs(mn) + 1e-12)).round(4),  # coefficient of variation
    })


# ── Quick diagnostics ─────────────────────────────────────────────────────────

def print_spatial_summary(label: str, moran_result: dict,
                           lisa_df: pd.DataFrame,
                           gi_df: pd.DataFrame) -> None:
    """Print a compact spatial diagnostics summary block."""
    quad = lisa_df["quadrant"].value_counts()
    hot  = gi_df["hotspot_class"].str.startswith("Hotspot").sum()
    cold = gi_df["hotspot_class"].str.startswith("Coldspot").sum()

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Global Moran's I : {moran_result['I']:.4f}  "
          f"(z={moran_result['z_score']:.2f}, p={moran_result['p_value_sim']:.4f})")
    print(f"  LISA clusters    : HH={quad.get('HH', 0)}  LL={quad.get('LL', 0)}  "
          f"HL={quad.get('HL', 0)}  LH={quad.get('LH', 0)}  NS={quad.get('NS', 0)}")
    print(f"  Getis-Ord Gi*    : {hot} hotspot(s),  {cold} coldspot(s)")
    print(f"{'─'*60}")
