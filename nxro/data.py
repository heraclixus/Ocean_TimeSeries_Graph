from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import xarray as xr
import warnings


class MonthlySequenceDataset(Dataset):
    """Pairs (X_t, t_years) -> X_{t+1}, and optionally multi-step targets up to rollout_k.

    Time is encoded as decimal years for Fourier seasonal embedding.
    """

    def __init__(self, ds: xr.Dataset, var_order: Optional[list] = None, rollout_k: int = 1):
        if var_order is None:
            var_order = list(ds.data_vars)
        # Assemble [time, n_vars]
        arrs = [ds[v].values for v in var_order]
        X = np.stack(arrs, axis=-1)
        # Build decimal years from time coords
        # Assumes monthly start-of-month timestamps
        time = ds.time.to_index()
        years = time.year + (time.month - 1) / 12.0
        T = X.shape[0]
        self.rollout_k = max(1, int(rollout_k))
        N = max(0, T - self.rollout_k)
        # Inputs (t=0..T-K-1) and one-step targets (t+1)
        X_in = X[:N].astype(np.float32)
        X_out = X[1:N + 1].astype(np.float32)
        t_in = years[:N].astype(np.float32)
        # Multi-step targets [N, K, V] when K>1
        if self.rollout_k > 1:
            seq = []
            for j in range(1, self.rollout_k + 1):
                seq.append(X[j:N + j])
            X_out_seq = np.stack(seq, axis=1).astype(np.float32)  # [N, K, V]
        else:
            X_out_seq = None

        # Drop any rows containing NaNs in inputs/targets (robust for extra training data)
        if N > 0:
            valid = np.isfinite(X_in).all(axis=1) & np.isfinite(X_out).all(axis=1)
            if X_out_seq is not None:
                valid = valid & np.isfinite(X_out_seq).all(axis=(1, 2))
            if not np.all(valid):
                X_in = X_in[valid]
                X_out = X_out[valid]
                t_in = t_in[valid]
                if X_out_seq is not None:
                    X_out_seq = X_out_seq[valid]

        self.X_in = X_in
        self.X_out = X_out
        self.t_in = t_in
        self.X_out_seq = X_out_seq

    def __len__(self) -> int:
        return self.X_in.shape[0]

    def __getitem__(self, idx: int):
        if self.X_out_seq is None:
            return (
                torch.from_numpy(self.X_in[idx]),
                torch.tensor(self.t_in[idx]),
                torch.from_numpy(self.X_out[idx]),
            )
        else:
            return (
                torch.from_numpy(self.X_in[idx]),
                torch.tensor(self.t_in[idx]),
                torch.from_numpy(self.X_out[idx]),
                torch.from_numpy(self.X_out_seq[idx]),
            )


def _load_obs(path: str) -> xr.Dataset:
    if path.endswith('.nc'):
        ds = xr.open_dataset(path)
    elif path.endswith('.csv'):
        df = xr.Dataset.from_dataframe(
            xr.DataArray.from_series(xr.DataArray(np.nan))
        )  # placeholder to avoid CSV for now
        raise NotImplementedError("CSV loading not implemented; use NetCDF")
    else:
        ds = xr.open_dataset(path)
    return ds


def get_dataloaders(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_slice: Tuple[str, str] = ('1979-01', '2022-12'),
    test_slice: Tuple[str, Optional[str]] = ('2023-01', None),
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    rollout_k: int = 1,
    extra_train_nc_paths: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, list]:
    """Create training and test dataloaders.

    The training dataset can optionally include additional NetCDF files. The
    test dataset always comes from the base ``nc_path``.
    """
    base_ds = _load_obs(nc_path)
    # canonical variable order from base dataset
    var_order = list(base_ds.data_vars)

    base_train = base_ds.sel(time=slice(train_slice[0], train_slice[1]))
    # Build list of per-source training datasets
    train_datasets: List[Dataset] = [
        MonthlySequenceDataset(base_train, var_order=var_order, rollout_k=rollout_k)
    ]

    # Optionally append extra training sources
    if extra_train_nc_paths:
        for p in extra_train_nc_paths:
            try:
                extra_ds = _load_obs(p)
            except Exception as e:
                warnings.warn(f"Failed to open extra training file '{p}': {e}")
                continue
            # Ensure variables match base var_order
            missing = [v for v in var_order if v not in extra_ds.data_vars]
            if missing:
                warnings.warn(
                    f"Skipping '{p}' because it lacks variables: {missing}. Expected {var_order}."
                )
                continue
            extra_train = extra_ds.sel(time=slice(train_slice[0], train_slice[1]))
            # align to base var order
            extra_train = extra_train[var_order]
            train_datasets.append(MonthlySequenceDataset(extra_train, var_order=var_order, rollout_k=rollout_k))

    # Compose final training dataset
    if len(train_datasets) == 1:
        dset_train = train_datasets[0]
    else:
        dset_train = ConcatDataset(train_datasets)

    # Test set strictly from base dataset
    if test_slice[1] is None:
        test_ds = base_ds.sel(time=slice(test_slice[0], None))
    else:
        test_ds = base_ds.sel(time=slice(test_slice[0], test_slice[1]))
    dset_test = MonthlySequenceDataset(test_ds, var_order=var_order, rollout_k=rollout_k)

    dl_train = DataLoader(dset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dl_test = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_train, dl_test, var_order


