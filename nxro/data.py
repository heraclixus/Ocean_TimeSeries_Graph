from typing import Tuple, Optional, List
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import xarray as xr
import warnings


# =============================================================================
# CESM2-LENS Climate Mode Data Configuration
# =============================================================================

# Variable name mapping from CESM2-LENS to standard NXRO names
CESM2_TO_NXRO_VAR_MAPPING = {
    'ENSO': 'Nino34',   # El Nino Southern Oscillation -> Nino3.4 SSTA
    'd20': 'WWV',       # Thermocline depth anomaly -> Warm Water Volume
    'NPMM': 'NPMM',     # North Pacific Meridional Mode
    'SPMM': 'SPMM',     # South Pacific Meridional Mode  
    'IOB': 'IOB',       # Indian Ocean Basin
    'TNA': 'TNA',       # Tropical North Atlantic
    'ALT3': 'ATL3',     # Atlantic Nino 3 (CESM2 uses ALT3, ORAS5 uses ATL3)
    'IOD': 'IOD',       # Indian Ocean Dipole
    'SIOD': 'SIOD',     # Subtropical Indian Ocean Dipole
    'SASD': 'SASD',     # South Atlantic Subtropical Dipole
}

# Default time slice for CESM2-LENS data (1979-2024 as requested)
CESM2_DEFAULT_TIME_SLICE = ('1979-01', '2024-12')

# Directory containing CESM2-LENS climate mode files
CESM2_CLIMATE_MODE_DIR = 'data/CESM2-LENS_climate_mode_data'

# Variables to use from CESM2-LENS (in order)
# This defines the canonical variable order for the model
# Set to None to use all available variables after mapping
CESM2_TARGET_VARS = None  # Use all available variables to match ORAS5 dimensions


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
        years = np.asarray(time.year + (time.month - 1) / 12.0, dtype=np.float32)
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


class MemorySequenceDataset(Dataset):
    """History-window dataset for memory-aware models.

    Each sample provides a window ``[X_{t-P}, ..., X_t]`` and the next state
    ``X_{t+1}``, with optional multi-step rollout targets.
    """

    def __init__(self, ds: xr.Dataset, var_order: Optional[list] = None,
                 rollout_k: int = 1, memory_depth: int = 1):
        if var_order is None:
            var_order = list(ds.data_vars)
        if memory_depth < 1:
            raise ValueError(f"memory_depth must be >= 1, got {memory_depth}")

        arrs = [ds[v].values for v in var_order]
        X = np.stack(arrs, axis=-1)
        time = ds.time.to_index()
        years = np.asarray(time.year + (time.month - 1) / 12.0, dtype=np.float32)
        T = X.shape[0]

        self.rollout_k = max(1, int(rollout_k))
        self.memory_depth = int(memory_depth)
        n_samples = max(0, T - self.rollout_k - self.memory_depth)

        if n_samples == 0:
            self.X_hist = np.zeros((0, self.memory_depth + 1, X.shape[1]), dtype=np.float32)
            self.t_hist = np.zeros((0, self.memory_depth + 1), dtype=np.float32)
            self.X_out = np.zeros((0, X.shape[1]), dtype=np.float32)
            self.X_out_seq = None if self.rollout_k == 1 else np.zeros((0, self.rollout_k, X.shape[1]), dtype=np.float32)
            return

        x_hist = []
        t_hist = []
        x_out = []
        x_out_seq = [] if self.rollout_k > 1 else None

        for idx in range(n_samples):
            current_idx = self.memory_depth + idx
            hist_slice = slice(current_idx - self.memory_depth, current_idx + 1)
            x_hist.append(X[hist_slice])
            t_hist.append(years[hist_slice])
            x_out.append(X[current_idx + 1])
            if x_out_seq is not None:
                x_out_seq.append(X[current_idx + 1: current_idx + 1 + self.rollout_k])

        X_hist = np.asarray(x_hist, dtype=np.float32)
        t_hist_arr = np.asarray(t_hist, dtype=np.float32)
        X_out = np.asarray(x_out, dtype=np.float32)
        X_out_seq = None if x_out_seq is None else np.asarray(x_out_seq, dtype=np.float32)

        valid = np.isfinite(X_hist).all(axis=(1, 2))
        valid = valid & np.isfinite(t_hist_arr).all(axis=1)
        valid = valid & np.isfinite(X_out).all(axis=1)
        if X_out_seq is not None:
            valid = valid & np.isfinite(X_out_seq).all(axis=(1, 2))

        if not np.all(valid):
            X_hist = X_hist[valid]
            t_hist_arr = t_hist_arr[valid]
            X_out = X_out[valid]
            if X_out_seq is not None:
                X_out_seq = X_out_seq[valid]

        self.X_hist = X_hist
        self.t_hist = t_hist_arr
        self.X_out = X_out
        self.X_out_seq = X_out_seq

    def __len__(self) -> int:
        return self.X_hist.shape[0]

    def __getitem__(self, idx: int):
        if self.X_out_seq is None:
            return (
                torch.from_numpy(self.X_hist[idx]),
                torch.from_numpy(self.t_hist[idx]),
                torch.from_numpy(self.X_out[idx]),
            )
        return (
            torch.from_numpy(self.X_hist[idx]),
            torch.from_numpy(self.t_hist[idx]),
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


def load_cesm2_climate_mode_file(
    path: str,
    target_vars: Optional[List[str]] = None,
    time_slice: Optional[Tuple[str, str]] = None,
    apply_default_time_slice: bool = True,
) -> xr.Dataset:
    """Load a single CESM2-LENS climate mode NC file and convert to NXRO format.
    
    Args:
        path: Path to the CESM2-LENS climate mode NC file.
        target_vars: List of target variable names (in NXRO format, e.g., ['Nino34', 'WWV']).
                     If None, uses CESM2_TARGET_VARS.
        time_slice: Optional (start, end) tuple for time slicing.
                    If None and apply_default_time_slice=True, uses CESM2_DEFAULT_TIME_SLICE.
        apply_default_time_slice: If True and time_slice is None, apply default time slice.
                                  Set to False to load full time range.
    
    Returns:
        xr.Dataset with renamed variables and optional time slicing.
    """
    ds = xr.open_dataset(path)
    
    # Unit conversion: d20 (CESM2) is in centimeters, WWV (ORAS5) is in meters
    # Convert d20 from cm to m before renaming
    if 'd20' in ds.data_vars:
        ds['d20'] = ds['d20'] / 100.0  # cm -> m
        print(f"  Converted d20 from cm to m (divided by 100)")
    
    # Rename variables from CESM2 names to NXRO names
    rename_dict = {}
    for cesm_name, nxro_name in CESM2_TO_NXRO_VAR_MAPPING.items():
        if cesm_name in ds.data_vars:
            rename_dict[cesm_name] = nxro_name
    
    if rename_dict:
        ds = ds.rename(rename_dict)
    
    # Select target variables if specified
    if target_vars is None:
        # If CESM2_TARGET_VARS is also None, use all available variables
        if CESM2_TARGET_VARS is None:
            # Use all variables after renaming (all mapped variables)
            available_vars = list(ds.data_vars)
        else:
            target_vars = CESM2_TARGET_VARS
            available_vars = [v for v in target_vars if v in ds.data_vars]
    else:
        # Filter to only include requested variables that exist
        available_vars = [v for v in target_vars if v in ds.data_vars]
    
    if not available_vars:
        raise ValueError(f"No target variables found in {path}. Available: {list(ds.data_vars)}")
    
    ds = ds[available_vars]
    
    # Apply time slicing
    if time_slice is not None:
        ds = ds.sel(time=slice(time_slice[0], time_slice[1]))
    elif apply_default_time_slice:
        ds = ds.sel(time=slice(CESM2_DEFAULT_TIME_SLICE[0], CESM2_DEFAULT_TIME_SLICE[1]))
    # else: no time slicing, return full dataset
    
    return ds


def discover_cesm2_climate_mode_files(
    base_dir: Optional[str] = None,
) -> List[str]:
    """Discover all CESM2-LENS climate mode NC files in a directory.
    
    Args:
        base_dir: Directory to search. If None, uses CESM2_CLIMATE_MODE_DIR.
    
    Returns:
        List of paths to climate mode NC files.
    """
    if base_dir is None:
        base_dir = CESM2_CLIMATE_MODE_DIR
    
    if not os.path.isdir(base_dir):
        warnings.warn(f"CESM2-LENS climate mode directory not found: {base_dir}")
        return []
    
    # Find all climate mode NC files
    pattern = os.path.join(base_dir, '*.climate-modes.*.nc')
    files = sorted(glob.glob(pattern))
    
    if not files:
        # Try alternative pattern
        pattern = os.path.join(base_dir, '*.nc')
        files = sorted(glob.glob(pattern))
    
    return files


def is_cesm2_climate_mode_file(path: str) -> bool:
    """Check if a path points to a CESM2-LENS climate mode file.
    
    Args:
        path: Path to check.
    
    Returns:
        True if the file is a CESM2-LENS climate mode file.
    """
    if not path.endswith('.nc'):
        return False
    
    # Check by directory
    if CESM2_CLIMATE_MODE_DIR in path:
        return True
    
    # Check by filename pattern
    basename = os.path.basename(path)
    if 'climate-modes' in basename.lower():
        return True
    if basename.startswith('b.e21.') and 'pop.h.' in basename:
        return True
    
    return False


def _load_obs_with_cesm2_support(
    path: str,
    target_vars: Optional[List[str]] = None,
    time_slice: Optional[Tuple[str, str]] = None,
) -> xr.Dataset:
    """Load observation file with automatic CESM2-LENS support.
    
    If the file is detected as CESM2-LENS climate mode data, it will be
    automatically converted to NXRO variable naming convention.
    
    Args:
        path: Path to NetCDF file.
        target_vars: Target variable names for CESM2-LENS files.
        time_slice: Time slice for CESM2-LENS files.
    
    Returns:
        xr.Dataset with standardized variable names.
    """
    if is_cesm2_climate_mode_file(path):
        return load_cesm2_climate_mode_file(path, target_vars=target_vars, time_slice=time_slice)
    else:
        return _load_obs(path)


def get_common_vars(cesm2_path: str, oras5_path: str, exclude_vars: Optional[List[str]] = None) -> List[str]:
    """Get the common variables between CESM2-LENS and ORAS5 datasets.
    
    This is essential for two-stage training to ensure both stages use
    the same variable set.
    
    Args:
        cesm2_path: Path to CESM2-LENS climate mode file.
        oras5_path: Path to ORAS5 observation file.
        exclude_vars: Optional list of variables to exclude.
    
    Returns:
        List of common variable names.
    """
    # Load CESM2-LENS and get variable names (after renaming)
    cesm2_ds = load_cesm2_climate_mode_file(cesm2_path, target_vars=None, time_slice=None)
    cesm2_vars = set(cesm2_ds.data_vars)
    cesm2_ds.close()
    
    # Load ORAS5 and get variable names
    oras5_ds = _load_obs(oras5_path)
    oras5_vars = set(oras5_ds.data_vars)
    oras5_ds.close()
    
    # Find common variables
    common = cesm2_vars & oras5_vars
    
    # Apply exclusions
    if exclude_vars:
        common = common - set(exclude_vars)
    
    # Return as sorted list to ensure consistent ordering
    common_list = sorted(list(common))
    
    print(f"  CESM2-LENS vars: {sorted(cesm2_vars)}")
    print(f"  ORAS5 vars: {sorted(oras5_vars)}")
    print(f"  Common vars ({len(common_list)}): {common_list}")
    
    return common_list


def get_dataloaders(
    nc_path: str = 'data/XRO_indices_oras5.nc',
    train_slice: Tuple[str, str] = ('1979-01', '2022-12'),
    test_slice: Tuple[str, Optional[str]] = ('2023-01', None),
    val_slice: Optional[Tuple[str, str]] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    rollout_k: int = 1,
    memory_depth: int = 0,
    extra_train_nc_paths: Optional[List[str]] = None,
    exclude_vars: Optional[List[str]] = None,
    cesm2_target_vars: Optional[List[str]] = None,
    include_only_vars: Optional[List[str]] = None,
):
    """Create training, (optional) validation, and test dataloaders.

    Returns (dl_train, dl_test, var_order) when val_slice is None, or
    (dl_train, dl_val, dl_test, var_order) when val_slice is provided.

    Args:
        nc_path: Path to the primary NetCDF file (e.g., ORAS5 observations).
        train_slice: (start, end) dates for training data.
        test_slice: (start, end) dates for test data.
        val_slice: Optional (start, end) dates for validation data.
                   When provided, early stopping should use val instead of test.
        batch_size: Batch size for dataloaders.
        shuffle: Whether to shuffle training data.
        num_workers: Number of dataloader workers.
        rollout_k: K-step rollout for sequence targets.
        memory_depth: If > 0, return history windows of length ``memory_depth + 1``.
        extra_train_nc_paths: Additional NC files for training (supports CESM2-LENS).
        exclude_vars: Optional list of variable names to exclude from the dataset.
        cesm2_target_vars: Target variable names when loading CESM2-LENS files.
        include_only_vars: If specified, filter dataset to only include these variables.
    """
    # Check if primary nc_path is a CESM2-LENS file
    if is_cesm2_climate_mode_file(nc_path):
        base_ds = load_cesm2_climate_mode_file(
            nc_path, 
            target_vars=cesm2_target_vars,
            time_slice=None,  # Don't slice yet, let train_slice/test_slice handle it
            apply_default_time_slice=False,  # Let train_slice/test_slice handle time filtering
        )
    else:
        base_ds = _load_obs(nc_path)
    
    # Filter to include only specified variables (for two-stage training compatibility)
    if include_only_vars:
        available = [v for v in include_only_vars if v in base_ds.data_vars]
        if not available:
            raise ValueError(f"None of the requested include_only_vars {include_only_vars} found in dataset. "
                           f"Available: {list(base_ds.data_vars)}")
        base_ds = base_ds[available]
    
    # Exclude specified variables if requested
    if exclude_vars:
        for var in exclude_vars:
            if var in base_ds.data_vars:
                base_ds = base_ds.drop_vars(var)
                
    # canonical variable order from base dataset
    var_order = list(base_ds.data_vars)
    dataset_cls = MemorySequenceDataset if memory_depth > 0 else MonthlySequenceDataset
    dataset_kwargs = {'var_order': var_order, 'rollout_k': rollout_k}
    if memory_depth > 0:
        dataset_kwargs['memory_depth'] = memory_depth

    base_train = base_ds.sel(time=slice(train_slice[0], train_slice[1]))
    # Build list of per-source training datasets
    train_datasets: List[Dataset] = [
        dataset_cls(base_train, **dataset_kwargs)
    ]

    # Optionally append extra training sources
    if extra_train_nc_paths:
        for p in extra_train_nc_paths:
            try:
                # Check if this is a CESM2-LENS climate mode file
                if is_cesm2_climate_mode_file(p):
                    # Use CESM2 loader with variable mapping
                    # Use base var_order as target vars for CESM2
                    extra_ds = load_cesm2_climate_mode_file(
                        p, 
                        target_vars=cesm2_target_vars or var_order,
                        time_slice=train_slice,
                    )
                else:
                    extra_ds = _load_obs(p)
            except Exception as e:
                warnings.warn(f"Failed to open extra training file '{p}': {e}")
                continue
            
            # Exclude specified variables from extra datasets too
            if exclude_vars:
                for var in exclude_vars:
                    if var in extra_ds.data_vars:
                        extra_ds = extra_ds.drop_vars(var)
            
            # Ensure variables match base var_order
            missing = [v for v in var_order if v not in extra_ds.data_vars]
            if missing:
                warnings.warn(
                    f"Skipping '{p}' because it lacks variables: {missing}. Expected {var_order}."
                )
                continue
            
            # For non-CESM2 files, apply time slice
            if not is_cesm2_climate_mode_file(p):
                extra_ds = extra_ds.sel(time=slice(train_slice[0], train_slice[1]))
            
            # align to base var order
            extra_train = extra_ds[var_order]
            train_datasets.append(dataset_cls(extra_train, **dataset_kwargs))

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
    dset_test = dataset_cls(test_ds, **dataset_kwargs)

    dl_train = DataLoader(dset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dl_test = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Optional validation set
    if val_slice is not None:
        val_ds_xr = base_ds.sel(time=slice(val_slice[0], val_slice[1]))
        dset_val = dataset_cls(val_ds_xr, **dataset_kwargs)
        dl_val = DataLoader(dset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return dl_train, dl_val, dl_test, var_order

    return dl_train, dl_test, var_order
