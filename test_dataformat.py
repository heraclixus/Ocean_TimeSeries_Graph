import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def _extract_year_month(time_values) -> tuple:
    """Return (years, months) arrays for either numpy datetime64 or cftime objects."""
    # Try numpy datetime64 via pandas
    try:
        if hasattr(time_values, 'dtype') and np.issubdtype(time_values.dtype, np.datetime64):
            idx = pd.to_datetime(time_values)
            return np.asarray(idx.year), np.asarray(idx.month)
    except Exception:
        pass
    # Fallback: iterate (works for cftime objects)
    years = np.array([t.year for t in list(time_values)])
    months = np.array([t.month for t in list(time_values)])
    return years, months


def months_step_series_from_values(time_values) -> np.ndarray:
    years, months = _extract_year_month(time_values)
    if years.size < 2:
        return np.array([], dtype=int)
    codes = years * 12 + months
    return codes[1:] - codes[:-1]


def compare_time_coords(base: xr.Dataset, cand: xr.Dataset, errors: list, warnings: list) -> None:
    if 'time' not in base.coords or 'time' not in cand.coords:
        errors.append("Missing 'time' coordinate in one of the datasets")
        return

    base_vals = base.time.values
    cand_vals = cand.time.values

    b_years, b_months = _extract_year_month(base_vals)
    c_years, c_months = _extract_year_month(cand_vals)

    if b_years.size != c_years.size:
        warnings.append(f"Time length differs: base={b_years.size} vs cand={c_years.size}")
    else:
        same_pairs = np.all((b_years == c_years) & (b_months == c_months))
        if not same_pairs:
            warnings.append('Time (year-month) differs between base and candidate (same length).')

    # Check strictly monthly step
    base_steps = months_step_series_from_values(base_vals)
    cand_steps = months_step_series_from_values(cand_vals)
    if base_steps.size > 0 and not np.all(base_steps == 1):
        warnings.append('Base time step is not strictly monthly (detected non-1 month deltas).')
    if cand_steps.size > 0 and not np.all(cand_steps == 1):
        warnings.append('Candidate time step is not strictly monthly (detected non-1 month deltas).')


def compare_variables(base: xr.Dataset, cand: xr.Dataset, errors: list, warnings: list) -> None:
    base_vars = set(base.data_vars)
    cand_vars = set(cand.data_vars)

    missing_in_cand = base_vars - cand_vars
    extra_in_cand = cand_vars - base_vars
    if missing_in_cand:
        errors.append(f"Missing variables in candidate: {sorted(list(missing_in_cand))}")
    if extra_in_cand:
        warnings.append(f"Extra variables in candidate: {sorted(list(extra_in_cand))}")

    common = sorted(list(base_vars & cand_vars))
    # Order check
    base_order = list(base.data_vars)
    cand_order = list(cand.data_vars)
    if [v for v in base_order if v in common] != [v for v in cand_order if v in common]:
        warnings.append('Variable order differs between base and candidate.')

    for var in common:
        b = base[var]
        c = cand[var]
        if list(b.dims) != list(c.dims):
            errors.append(f"Dims differ for '{var}': base={list(b.dims)} vs cand={list(c.dims)}")
            continue
        # Compare sizes per dim
        for d in b.dims:
            if b.sizes[d] != c.sizes[d]:
                errors.append(f"Dim size mismatch for '{var}' dim '{d}': base={b.sizes[d]} vs cand={c.sizes[d]}")
        # Dtype
        if b.dtype != c.dtype:
            warnings.append(f"Dtype differs for '{var}': base={b.dtype} vs cand={c.dtype}")
        # Units attribute if present
        ub = b.attrs.get('units', None)
        uc = c.attrs.get('units', None)
        if (ub is not None or uc is not None) and ub != uc:
            warnings.append(f"Units differ for '{var}': base={ub} vs cand={uc}")
        # NaN presence
        b_nan = int(np.isnan(b.values).sum())
        c_nan = int(np.isnan(c.values).sum())
        if (b_nan == 0 and c_nan > 0) or (b_nan > 0 and c_nan == 0):
            warnings.append(f"NaN count differs for '{var}': base={b_nan} vs cand={c_nan}")


def compare_global_attrs(base: xr.Dataset, cand: xr.Dataset, warnings: list) -> None:
    # Non-fatal: note substantial differences in known attrs
    keys = sorted(set(list(base.attrs.keys()) + list(cand.attrs.keys())))
    diffs = []
    for k in keys:
        if base.attrs.get(k) != cand.attrs.get(k):
            diffs.append(k)
    if diffs:
        warnings.append(f"Global attrs differ on keys: {diffs}")


def check_format(base_path: str, candidate_path: str) -> bool:
    print(f"\n=== Checking candidate: {candidate_path} ===")
    ok = True
    try:
        base_ds = xr.open_dataset(base_path)
    except Exception as e:
        print(f"[ERROR] Failed to open base dataset '{base_path}': {e}")
        return False
    try:
        cand_ds = xr.open_dataset(candidate_path)
    except Exception as e:
        print(f"[ERROR] Failed to open candidate dataset '{candidate_path}': {e}")
        return False

    errors = []
    warnings_list = []
    compare_time_coords(base_ds, cand_ds, errors, warnings_list)
    compare_variables(base_ds, cand_ds, errors, warnings_list)
    compare_global_attrs(base_ds, cand_ds, warnings_list)

    if errors:
        ok = False
        print('[FAIL] Critical incompatibilities found:')
        for msg in errors:
            print(f"  - {msg}")
    else:
        print('[PASS] No critical format differences detected (dims/vars/time).')

    if warnings_list:
        print('[INFO] Non-critical differences:')
        for msg in warnings_list:
            print(f"  - {msg}")
    return ok


def auto_discover_candidates(data_dir: str, base_path: str) -> list:
    pattern = os.path.join(data_dir, 'XRO_indices_*.nc')
    all_files = sorted(glob.glob(pattern))
    base_abs = os.path.abspath(base_path)
    cand = [p for p in all_files if os.path.abspath(p) != base_abs]
    # Use only one candidate by default (generated by the same script)
    return cand[:1]


def main():
    parser = argparse.ArgumentParser(description='Verify XRO simulation NetCDF format matches ORAS5 indices file')
    parser.add_argument('--base', type=str, default='data/XRO_indices_oras5.nc', help='Base ORAS5 indices NetCDF path')
    parser.add_argument('--candidates', type=str, nargs='*', default=None, help='Candidate NetCDF paths to check')
    parser.add_argument('--data_dir', type=str, default='data', help='When --candidates empty, auto-discover in this directory')
    args = parser.parse_args()

    if args.candidates is None or len(args.candidates) == 0:
        candidates = auto_discover_candidates(args.data_dir, args.base)
        if not candidates:
            print('No candidate files found. Provide --candidates or add files matching data/XRO_indices_*.nc')
            sys.exit(1)
        print(f'Auto-discovered {len(candidates)} candidate(s). Using the first one for comparison.')
    else:
        candidates = args.candidates

    overall_ok = True
    for cand in candidates:
        ok = check_format(args.base, cand)
        overall_ok = overall_ok and ok

    if overall_ok:
        print('\nAll candidates are compatible with the base format.')
        sys.exit(0)
    else:
        print('\nSome candidates are not fully compatible. See messages above for guidance.')
        sys.exit(2)


if __name__ == '__main__':
    main()


