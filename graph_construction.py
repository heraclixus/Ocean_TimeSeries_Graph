import os
from typing import Tuple, Optional

import numpy as np
import torch
import xarray as xr
import pandas as pd

from XRO.core import XRO


def build_xro_coupling_graph(nc_path: str = 'data/XRO_indices_oras5.nc',
                             train_start: str = '1979-01', train_end: str = '2022-12',
                             var_order: list = None,
                             threshold: float = 0.0,
                             use_normalized: bool = True) -> Tuple[torch.Tensor, list]:
    """Construct an adjacency matrix from XRO's fitted linear seasonal operator.

    Steps:
    - Fit XRO on the base ORAS5 dataset over train period.
    - Extract Lac (seasonal linear operator over cycle) or its normalized version.
    - Aggregate over cycle by mean absolute value to get a static coupling strength.
    - Apply threshold to prune weak edges; keep self-loops optionally (we add them downstream).

    Returns:
        A (torch.Tensor): [V,V] adjacency strength (nonnegative), var_order list
    """
    ds = xr.open_dataset(nc_path)
    ds_train = ds.sel(time=slice(train_start, train_end))
    if var_order is None:
        var_order = list(ds_train.data_vars)
    # XRO expects X with shape [rank, time]
    X_np = np.stack([ds_train[v].values for v in var_order], axis=0)
    xro = XRO(ncycle=12, ac_order=2)
    fit_ds = xro.fit_matrix(X_np)
    if use_normalized:
        # Use normalized operator if available
        try:
            Lac_da = fit_ds['normLac']
        except Exception:
            Lac_da = fit_ds['Lac']
    else:
        Lac_da = fit_ds['Lac']
    # Aggregate over cycle: mean absolute coupling per (ranky, rankx)
    L_abs_mean = np.abs(Lac_da.values).mean(axis=2)  # [ranky, rankx]
    # Zero out diagonal for teleconnections; self-loops can be re-added during normalization
    np.fill_diagonal(L_abs_mean, 0.0)
    # Threshold
    if threshold > 0:
        L_abs_mean[L_abs_mean < threshold] = 0.0
    # Symmetrize by max to ensure undirected graph
    A = np.maximum(L_abs_mean, L_abs_mean.T)
    A_t = torch.tensor(A, dtype=torch.float32)
    return A_t, var_order


def build_corr_knn_graph(nc_path: str, train_start: str, train_end: str,
                         var_order: list, top_k: int = 2) -> torch.Tensor:
    import torch
    from nxro.models import build_edge_index_from_corr
    ds = xr.open_dataset(nc_path).sel(time=slice(train_start, train_end))
    X_np = np.stack([ds[v].values for v in var_order], axis=-1)
    X = torch.tensor(X_np, dtype=torch.float32)
    corr = torch.corrcoef(X.T)
    edge_index = build_edge_index_from_corr(corr, top_k=top_k)
    # Build adjacency from edge_index
    V = len(var_order)
    A = torch.zeros(V, V, dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    # Symmetrize and zero diagonal (will add self-loops later)
    A = torch.maximum(A, A.T)
    A.fill_diagonal_(0.0)
    return A


def normalize_with_self_loops(A: torch.Tensor) -> torch.Tensor:
    V = A.shape[0]
    A_sl = A.clone()
    A_sl = A_sl + torch.eye(V, dtype=A_sl.dtype, device=A_sl.device)
    rowsum = A_sl.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return A_sl / rowsum



def _safe_name(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def _hash_var_order(var_order: list) -> str:
    import hashlib
    joined = ','.join([str(v) for v in (var_order or [])])
    return hashlib.md5(joined.encode('utf-8')).hexdigest()[:8]


def _cache_path_for_xro_graph(nc_path: str, train_start: str, train_end: str,
                              var_order: list, threshold: float, use_normalized: bool,
                              cache_dir: str = 'results/graphs') -> str:
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{_safe_name(nc_path)}_{train_start}_{train_end}_norm{int(use_normalized)}_th{threshold:g}_vo{_hash_var_order(var_order)}"
    fname = f"xro_adj_{tag}.npz"
    return os.path.join(cache_dir, fname)


def get_or_build_xro_graph(nc_path: str = 'data/XRO_indices_oras5.nc',
                           train_start: str = '1979-01', train_end: str = '2022-12',
                           var_order: list = None,
                           threshold: float = 0.0,
                           use_normalized: bool = True,
                           cache_dir: str = 'results/graphs') -> Tuple[torch.Tensor, list]:
    """Return cached XRO-derived adjacency if present; otherwise build and cache it.

    The cache stores A (float32) and var_order (string list) for consistent indexing.
    If cached var_order differs from the requested one but is a permutation, A is reordered.
    """
    cache_path = _cache_path_for_xro_graph(nc_path, train_start, train_end, var_order or [], threshold, use_normalized, cache_dir)
    try:
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            A_np = data['A'].astype(np.float32)
            saved_order = list(data['var_order'].tolist()) if 'var_order' in data else (var_order or [])
            req_order = var_order or saved_order
            if saved_order and req_order and saved_order != req_order:
                if set(saved_order) == set(req_order) and len(saved_order) == len(req_order):
                    idx = [saved_order.index(v) for v in req_order]
                    A_np = A_np[np.ix_(idx, idx)]
                else:
                    # Incompatible; rebuild below
                    raise RuntimeError('Cached adjacency var_order incompatible; rebuilding')
            return torch.tensor(A_np, dtype=torch.float32), req_order
    except Exception:
        pass

    # Build and cache
    A_t, vo = build_xro_coupling_graph(nc_path=nc_path, train_start=train_start, train_end=train_end,
                                       var_order=var_order, threshold=threshold, use_normalized=use_normalized)
    try:
        np.savez(cache_path, A=A_t.cpu().numpy().astype(np.float32), var_order=np.array(vo, dtype=object))
    except Exception:
        pass
    return A_t, vo


# -------- Statistical (non-neural) interaction strength and KNN graph --------

def _load_series_matrix(data_path: str,
                        train_start: str,
                        train_end: str,
                        var_order: Optional[list]) -> Tuple[np.ndarray, list]:
    """Load [T,V] matrix from NetCDF or CSV within [train_start, train_end].

    - NetCDF: expects variables as data_vars, coordinate 'time'.
    - CSV: expects a 'time' column and variable columns; ignores 'month' if present.
    """
    if data_path.lower().endswith('.csv'):
        df = pd.read_csv(data_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            m = (df['time'] >= pd.to_datetime(train_start)) & (df['time'] <= (pd.to_datetime(train_end)))
            df = df.loc[m]
        cols = [c for c in df.columns if c not in ('time', 'month')]
        if var_order is None:
            var_order = cols
        else:
            var_order = [c for c in var_order if c in df.columns]
        X = df[var_order].to_numpy(dtype=np.float32)
        return X, var_order
    else:
        ds = xr.open_dataset(data_path).sel(time=slice(train_start, train_end))
        if var_order is None:
            var_order = list(ds.data_vars)
        X = np.stack([ds[v].values for v in var_order], axis=-1).astype(np.float32)  # [T,V]
        return X, var_order


def _standardize_2d(X: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    mu = np.nanmean(Xc, axis=0, keepdims=True)
    Xc = Xc - mu
    sd = np.nanstd(Xc, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return Xc / sd


def _pearson_strength(X: np.ndarray) -> np.ndarray:
    Xz = _standardize_2d(X)
    S = np.corrcoef(Xz, rowvar=False)
    S = np.nan_to_num(S, nan=0.0)
    return np.abs(S)


def _rankdata_vector(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata
        return rankdata(x, method='average').astype(np.float32)
    except Exception:
        # Fallback: simple stable ranks without tie-averaging
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(1, len(x) + 1, dtype=np.float32)
        return ranks


def _spearman_strength(X: np.ndarray) -> np.ndarray:
    T, V = X.shape
    R = np.zeros_like(X, dtype=np.float32)
    for j in range(V):
        col = X[:, j]
        mask = np.isfinite(col)
        r = np.empty_like(col, dtype=np.float32)
        r[~mask] = np.nan
        if mask.sum() > 0:
            r[mask] = _rankdata_vector(col[mask])
        else:
            r[:] = np.nan
        R[:, j] = r
    return _pearson_strength(R)


def _nmi_from_hist2d(x: np.ndarray, y: np.ndarray, n_bins: int = 16) -> float:
    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < max(10, n_bins):
        return 0.0
    # Equal-width bins on standardized variables
    xz = (x - x.mean()) / (x.std() + 1e-8)
    yz = (y - y.mean()) / (y.std() + 1e-8)
    # Clip to reasonable range to avoid extreme tails dominating bins
    xz = np.clip(xz, -5.0, 5.0)
    yz = np.clip(yz, -5.0, 5.0)
    H, xedges, yedges = np.histogram2d(xz, yz, bins=n_bins)
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_arg = Pxy / (Px * Py)
        log_arg[~np.isfinite(log_arg)] = 1.0
        MI = (Pxy * np.log(log_arg)).sum()
        Hx = -(Px * np.log(np.where(Px > 0, Px, 1.0))).sum()
        Hy = -(Py * np.log(np.where(Py > 0, Py, 1.0))).sum()
    denom = np.sqrt(max(Hx, 1e-12) * max(Hy, 1e-12))
    if denom <= 0:
        return 0.0
    nmi = float(MI / denom)
    return max(0.0, min(1.0, nmi))


def _mi_strength(X: np.ndarray, n_bins: int = 16) -> np.ndarray:
    T, V = X.shape
    S = np.zeros((V, V), dtype=np.float32)
    for i in range(V):
        S[i, i] = 1.0
        for j in range(i + 1, V):
            v = _nmi_from_hist2d(X[:, i], X[:, j], n_bins=n_bins)
            S[i, j] = S[j, i] = v
    return S


def _max_xcorr_strength(X: np.ndarray, max_lag: int = 3) -> np.ndarray:
    Xz = _standardize_2d(X)
    T, V = Xz.shape
    S = np.zeros((V, V), dtype=np.float32)
    for i in range(V):
        S[i, i] = 1.0
        xi = Xz[:, i]
        for j in range(i + 1, V):
            xj = Xz[:, j]
            best = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    a, b = xi, xj
                elif lag > 0:
                    a, b = xi[lag:], xj[:-lag]
                else:
                    a, b = xi[:lag], xj[-lag:]
                if len(a) < 10:
                    continue
                c = np.corrcoef(a, b)[0, 1]
                if not np.isfinite(c):
                    c = 0.0
                best = max(best, abs(float(c)))
            S[i, j] = S[j, i] = best
    return S


def _knn_from_strength(S: np.ndarray, top_k: int) -> np.ndarray:
    V = S.shape[0]
    A = np.zeros((V, V), dtype=np.float32)
    np.fill_diagonal(S, 0.0)
    for i in range(V):
        idx = np.argpartition(-S[i], kth=min(top_k, V - 1) - 1)[:top_k]
        for j in idx:
            if i != j and S[i, j] > 0:
                A[i, j] = S[i, j]
    A = np.maximum(A, A.T)
    return A


def _cache_path_for_stat_graph(data_path: str, train_start: str, train_end: str,
                               var_order: list, method: str, top_k: int,
                               n_bins: int, max_lag: int,
                               cache_dir: str = 'results/graphs') -> str:
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"{_safe_name(data_path)}_{train_start}_{train_end}_{method}_k{top_k}_b{n_bins}_lag{max_lag}_vo{_hash_var_order(var_order)}"
    fname = f"stat_adj_{tag}.npz"
    return os.path.join(cache_dir, fname)


def get_or_build_stat_knn_graph(data_path: str,
                                train_start: str,
                                train_end: str,
                                var_order: Optional[list] = None,
                                method: str = 'pearson',
                                top_k: int = 2,
                                n_bins: int = 16,
                                max_lag: int = 3,
                                cache_dir: str = 'results/graphs') -> Tuple[torch.Tensor, list]:
    """KNN adjacency from whole-sequence statistical interaction strength.

    method ∈ {'pearson','spearman','mi','xcorr_max'}; all are non-neural.
    - pearson: |corr| over entire sequence
    - spearman: |Spearman ρ| via ranks
    - mi: normalized mutual information via binned histogram (n_bins)
    - xcorr_max: max |corr| over lags in [-max_lag, max_lag]
    """
    # Try cache
    cache_path = _cache_path_for_stat_graph(data_path, train_start, train_end, var_order or [], method, top_k, n_bins, max_lag, cache_dir)
    try:
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            A_np = data['A'].astype(np.float32)
            saved_order = list(data['var_order'].tolist()) if 'var_order' in data else (var_order or [])
            req_order = var_order or saved_order
            if saved_order and req_order and saved_order != req_order:
                if set(saved_order) == set(req_order) and len(saved_order) == len(req_order):
                    idx = [saved_order.index(v) for v in req_order]
                    A_np = A_np[np.ix_(idx, idx)]
                else:
                    raise RuntimeError('Cached adjacency var_order incompatible; rebuilding')
            return torch.tensor(A_np, dtype=torch.float32), req_order
    except Exception:
        pass

    # Build
    X, vo = _load_series_matrix(data_path, train_start, train_end, var_order)
    if method == 'pearson':
        S = _pearson_strength(X)
    elif method == 'spearman':
        S = _spearman_strength(X)
    elif method == 'mi':
        S = _mi_strength(X, n_bins=n_bins)
    elif method == 'xcorr_max':
        S = _max_xcorr_strength(X, max_lag=max_lag)
    else:
        raise ValueError(f"Unknown method: {method}")
    A = _knn_from_strength(S, top_k=top_k)

    try:
        np.savez(cache_path, A=A.astype(np.float32), var_order=np.array(vo, dtype=object))
    except Exception:
        pass
    return torch.tensor(A, dtype=torch.float32), vo

