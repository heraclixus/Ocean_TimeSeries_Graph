import argparse
import os
import glob
from typing import Dict, List

import numpy as np
import xarray as xr


def rename_variables(ds: xr.Dataset, mapping: Dict[str, str]) -> xr.Dataset:
    to_rename = {src: dst for src, dst in mapping.items() if src in ds.data_vars}
    if to_rename:
        ds = ds.rename(to_rename)
    return ds


def align_wvv_from_base(base: xr.Dataset, cand: xr.Dataset, var_name: str = 'WWV') -> xr.DataArray:
    if var_name not in base.data_vars:
        raise ValueError(f"Base dataset missing '{var_name}' variable.")
    if 'time' not in cand.coords:
        raise ValueError("Candidate dataset missing 'time' coordinate.")
    return align_var_by_year_month(base, var_name, cand.time)


def _year_month_from_time(tcoord: xr.DataArray):
    try:
        years = tcoord.dt.year.values
        months = tcoord.dt.month.values
    except Exception:
        vals = list(tcoord.values)
        years = np.array([v.year for v in vals])
        months = np.array([v.month for v in vals])
    return years, months


def align_var_by_year_month(base: xr.Dataset, var_name: str, target_time: xr.DataArray) -> xr.DataArray:
    by, bm = _year_month_from_time(base['time'])
    tv = base[var_name].values
    mapping = {(int(y), int(m)): tv[i] for i, (y, m) in enumerate(zip(by, bm))}
    ty, tm = _year_month_from_time(target_time)
    out = np.array([mapping.get((int(y), int(m)), np.nan) for y, m in zip(ty, tm)], dtype=base[var_name].dtype)
    da = xr.DataArray(out, coords={'time': target_time}, dims=['time'])
    da.attrs.update(base[var_name].attrs)
    return da


def standardize_units_and_dtype(base: xr.Dataset, ds: xr.Dataset, vars_to_fix: List[str]) -> xr.Dataset:
    fixed = ds
    for v in vars_to_fix:
        if v not in fixed.data_vars:
            continue
        # Units: copy from base if present
        units = base[v].attrs.get('units', None)
        if units is not None:
            fixed[v].attrs['units'] = units
        # Dtype: cast to base dtype if different
        b_dt = base[v].dtype
        if fixed[v].dtype != b_dt:
            try:
                fixed[v] = fixed[v].astype(b_dt)
            except Exception:
                pass
    return fixed


def preprocess(base_path: str, input_path: str, output_path: str, nan_ratio_threshold: float = 0.5) -> None:
    print(f"Base:   {base_path}")
    print(f"Input:  {input_path}")
    base = xr.open_dataset(base_path)
    ds = xr.open_dataset(input_path)

    # 1) Apply requested renames
    rename_map = {
        'ALT3': 'ATL3',
        'ENSO': 'Nino34',
    }
    before_vars = list(ds.data_vars)
    ds = rename_variables(ds, rename_map)
    after_vars = list(ds.data_vars)
    renamed = [f"{k}->{v}" for k, v in rename_map.items() if k in before_vars]
    if renamed:
        print("Renamed:", ", ".join(renamed))

    # 2) Ensure WWV present: inject from base aligned to candidate time
    added = []
    if 'WWV' not in ds.data_vars:
        try:
            wvv = align_wvv_from_base(base, ds, 'WWV')
            ds = ds.assign(WWV=wvv)
            added.append('WWV')
        except Exception as e:
            print(f"[WARN] Could not inject WWV from base: {e}")

    # 3) Drop extra variables not in base; keep only base variable set
    base_vars = list(base.data_vars)
    extra = [v for v in ds.data_vars if v not in base_vars]
    if extra:
        ds = ds.drop_vars(extra)
        print("Dropped extras:", ", ".join(extra))

    # 4) Add any remaining missing variables from base as copies (aligned by year-month)
    missing = [v for v in base_vars if v not in ds.data_vars]
    for v in missing:
        try:
            ds = ds.assign({v: align_var_by_year_month(base, v, ds.time)})
            added.append(v)
        except Exception as e:
            print(f"[WARN] Could not add missing variable '{v}' from base: {e}")

    if added:
        print("Added from base:", ", ".join(sorted(set(added))))

    # 4.5) Replace variables with too many missing values by ORAS5 values aligned by time
    replaced = []
    if 'time' in ds.coords:
        T = int(ds.sizes.get('time', 0))
        for v in list(ds.data_vars):
            if v not in base_vars:
                continue
            try:
                nan_count = int(np.isnan(ds[v].values).sum())
                ratio = (nan_count / max(T, 1)) if T > 0 else 0.0
                if ratio > nan_ratio_threshold:
                    ds[v] = align_var_by_year_month(base, v, ds.time)
                    replaced.append((v, ratio))
            except Exception:
                continue
    if replaced:
        msg = ", ".join([f"{v} (NaN ratio {r:.2f})" for v, r in replaced])
        print(f"Replaced by ORAS5 due to high missing ratio (> {nan_ratio_threshold:.2f}): {msg}")

    # 5) Reorder variables to match base order
    ds = ds[base_vars]

    # 6) Standardize units attrs and dtypes to match base for all variables
    ds = standardize_units_and_dtype(base, ds, base_vars)

    # 7) Basic sanity logs
    if 'time' in base.coords and 'time' in ds.coords:
        if len(base.time) != len(ds.time):
            print(f"[INFO] Time length differs: base={len(base.time)} vs cand={len(ds.time)}")
        else:
            # Compare year-month pairs
            try:
                b_years = base['time'].dt.year.values
                b_months = base['time'].dt.month.values
                c_years = ds['time'].dt.year.values
                c_months = ds['time'].dt.month.values
                if not np.all((b_years == c_years) & (b_months == c_months)):
                    print("[INFO] Time (year-month) differs between base and candidate.")
            except Exception:
                pass

    # 8) Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"Wrote:  {output_path}")


def discover_inputs(base_path: str) -> List[str]:
    data_dir = os.path.dirname(base_path) or 'data'
    pattern = os.path.join(data_dir, 'XRO_indices_*.nc')
    all_files = sorted(glob.glob(pattern))
    base_abs = os.path.abspath(base_path)
    # Exclude base and already preprocessed files
    out = [p for p in all_files if os.path.abspath(p) != base_abs and not p.endswith('_preproc.nc')]
    return out


def main():
    parser = argparse.ArgumentParser(description='Preprocess climate simulation NetCDF to match ORAS5 indices format')
    parser.add_argument('--base', type=str, default='data/XRO_indices_oras5.nc', help='Path to ORAS5 indices NetCDF')
    parser.add_argument('--input', type=str, nargs='*', default=None,
                        help='Path(s) to climate simulation NetCDF(s) to preprocess. If omitted, auto-discovers all new files in data/.')
    parser.add_argument('--output', type=str, default=None, help='Output path for a single input (default: <input>_preproc.nc). Ignored when multiple inputs.')
    parser.add_argument('--nan_ratio_threshold', type=float, default=0.0,
                        help='If fraction of NaNs in a variable exceeds this, replace it from ORAS5 (default 0.2)')
    args = parser.parse_args()

    inputs = args.input
    if not inputs:
        inputs = discover_inputs(args.base)
        if not inputs:
            print('No candidate files found to preprocess. Place files named XRO_indices_*.nc under the data/ directory.')
            return
        print(f"Discovered {len(inputs)} file(s) to preprocess under {os.path.dirname(args.base) or 'data'}.")

    if len(inputs) == 1:
        in_path = inputs[0]
        output = args.output
        if output is None:
            root, ext = os.path.splitext(in_path)
            output = f"{root}_preproc{ext or '.nc'}"
        preprocess(args.base, in_path, output, nan_ratio_threshold=args.nan_ratio_threshold)
    else:
        for in_path in inputs:
            root, ext = os.path.splitext(in_path)
            output = f"{root}_preproc{ext or '.nc'}"
            preprocess(args.base, in_path, output, nan_ratio_threshold=args.nan_ratio_threshold)


if __name__ == '__main__':
    main()


