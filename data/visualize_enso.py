import os
import glob
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def _year_month_from_time(tcoord: xr.DataArray):
    try:
        years = tcoord.dt.year.values
        months = tcoord.dt.month.values
    except Exception:
        vals = list(tcoord.values)
        years = np.array([v.year for v in vals])
        months = np.array([v.month for v in vals])
    return years, months


def align_by_year_month(da_a: xr.DataArray, da_b: xr.DataArray):
    ya, ma = _year_month_from_time(da_a['time'])
    yb, mb = _year_month_from_time(da_b['time'])

    keys_a = list(zip(ya.astype(int), ma.astype(int)))
    keys_b = list(zip(yb.astype(int), mb.astype(int)))
    set_a = set(keys_a)
    set_b = set(keys_b)
    common = sorted(list(set_a & set_b))
    if len(common) == 0:
        return np.array([]), np.array([]), np.array([])

    map_a = {k: float(da_a.values[i]) for i, k in enumerate(keys_a)}
    map_b = {k: float(da_b.values[i]) for i, k in enumerate(keys_b)}
    xa = np.array([map_a.get(k, np.nan) for k in common], dtype=float)
    xb = np.array([map_b.get(k, np.nan) for k in common], dtype=float)
    # Build datetime index at month-start for plotting
    dates = pd.to_datetime([f"{y:04d}-{m:02d}-01" for (y, m) in common])
    return dates, xa, xb


def compute_metrics(x_ref: np.ndarray, x_cmp: np.ndarray):
    m = np.isfinite(x_ref) & np.isfinite(x_cmp)
    if m.sum() == 0:
        return np.nan, np.nan
    ref = x_ref[m]
    cmpv = x_cmp[m]
    # ACC
    if ref.std() == 0 or cmpv.std() == 0:
        acc = np.nan
    else:
        acc = np.corrcoef(ref, cmpv)[0, 1]
    rmse = float(np.sqrt(np.mean((ref - cmpv) ** 2)))
    return acc, rmse


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Visualize Nino34 from preprocessed simulation files vs ORAS5')
    parser.add_argument('--base', type=str, default='data/XRO_indices_oras5.nc', help='Path to ORAS5 indices NetCDF')
    parser.add_argument('--outdir', type=str, default='results/enso', help='Directory to save comparison plots')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    base_path = args.base
    if not os.path.exists(base_path):
        print(f"Base file not found: {base_path}")
        return
    base = xr.open_dataset(base_path)
    if 'Nino34' not in base.data_vars:
        print("Base dataset missing 'Nino34'.")
        return
    n34_base = base['Nino34']

    preproc_files = sorted(glob.glob('data/XRO_indices_*_preproc.nc'))
    if not preproc_files:
        print('No preprocessed files found under data/. Run preprocess_clim_data.py first.')
        return

    for p in preproc_files:
        try:
            ds = xr.open_dataset(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        if 'Nino34' not in ds.data_vars:
            print(f"Skipping {p}: missing 'Nino34'")
            continue
        n34_sim = ds['Nino34']

        dates, a, b = align_by_year_month(n34_base, n34_sim)
        if dates.size < 3:
            print(f"Skipping {p}: insufficient overlap after alignment")
            continue

        acc, rmse = compute_metrics(a, b)

        fig, ax = plt.subplots(1, 1, figsize=(11, 4))
        ax.plot(dates, a, color='black', label='ORAS5 Nino3.4')
        ax.plot(dates, b, color='tab:red', label='Sim Nino3.4')
        ax.set_title(f"Nino3.4 comparison: {os.path.basename(p)} | ACC={acc:.2f} RMSE={rmse:.2f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('SSTA (°C)')
        ax.legend()
        ax.grid(alpha=0.2)
        out_name = os.path.join(args.outdir, f"enso_compare_{os.path.splitext(os.path.basename(p))[0]}.png")
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close()
        print(f"Saved {out_name}")


if __name__ == '__main__':
    main()


