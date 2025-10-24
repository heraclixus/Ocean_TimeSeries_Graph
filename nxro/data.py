from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr


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
        self.X_in = X[:N].astype(np.float32)
        self.X_out = X[1:N + 1].astype(np.float32)
        self.t_in = years[:N].astype(np.float32)
        # Multi-step targets [N, K, V] when K>1
        if self.rollout_k > 1:
            seq = []
            for j in range(1, self.rollout_k + 1):
                seq.append(X[j:N + j])
            self.X_out_seq = np.stack(seq, axis=1).astype(np.float32)  # [N, K, V]
        else:
            self.X_out_seq = None

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
) -> Tuple[DataLoader, DataLoader, list]:
    ds = _load_obs(nc_path)
    # enforce variable order from dataset
    var_order = list(ds.data_vars)
    train_ds = ds.sel(time=slice(train_slice[0], train_slice[1]))
    if test_slice[1] is None:
        test_ds = ds.sel(time=slice(test_slice[0], None))
    else:
        test_ds = ds.sel(time=slice(test_slice[0], test_slice[1]))

    dset_train = MonthlySequenceDataset(train_ds, var_order=var_order, rollout_k=rollout_k)
    dset_test = MonthlySequenceDataset(test_ds, var_order=var_order, rollout_k=rollout_k)
    dl_train = DataLoader(dset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dl_test = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_train, dl_test, var_order


