import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import xarray as xr

from XRO.core import XRO
from utils.xro_utils import calc_forecast_skill


DEFAULT_NC_PATH = "data/XRO_indices_oras5.nc"
DEFAULT_TRAIN_PERIOD = ("1979-01", "2001-12")
DEFAULT_TEST_PERIOD = ("2002-01", "2022-12")
DEFAULT_FIT_PATH = "results_out_of_sample/xro_baseline/xro_fit.nc"
DEFAULT_SUMMARY_PATH = "results_out_of_sample/xro_baseline/xro_test_rmse_by_lead_current_eval_summary.json"
DEFAULT_CSV_PATH = "results_out_of_sample/xro_baseline/xro_test_rmse_by_lead_current_eval.csv"
HISTORICAL_LEADS = list(range(0, 22))
DIAGNOSTIC_LEADS = list(range(1, 21))


def _ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def _normalize_leads(leads: Optional[Iterable[int]]) -> List[int]:
    if leads is None:
        return list(range(1, 21))
    return [int(lead) for lead in leads]


def _load_cached_summary(summary_path: str, csv_path: str, leads: Sequence[int]) -> Optional[dict]:
    if not (os.path.exists(summary_path) and os.path.exists(csv_path)):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    cached_leads = summary.get("leads", [])
    if list(cached_leads) != list(leads):
        return None
    return summary


def _save_curve_csv(csv_path: str, leads: Sequence[int], rmse_by_lead: Sequence[float], acc_by_lead: Sequence[float]) -> None:
    _ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lead", "rmse_xro", "acc_xro"])
        for lead, rmse, acc in zip(leads, rmse_by_lead, acc_by_lead):
            writer.writerow([int(lead), float(rmse), float(acc)])


def get_xro_baseline_metrics(
    *,
    nc_path: str = DEFAULT_NC_PATH,
    train_period: Sequence[str] = DEFAULT_TRAIN_PERIOD,
    test_period: Sequence[str] = DEFAULT_TEST_PERIOD,
    fit_path: str = DEFAULT_FIT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
    leads: Optional[Iterable[int]] = None,
    label: str = "XRO",
    force_recompute: bool = False,
) -> dict:
    """Return cached or freshly computed XRO RMSE/ACC-by-lead on the ORAS5 OOS split.

    The computation uses the repo's current local evaluation path
    (`calc_forecast_skill`), so it stays consistent with the memory-model plots
    regenerated in the same environment.
    """

    lead_list = _normalize_leads(leads)
    if not force_recompute:
        cached = _load_cached_summary(summary_path, csv_path, lead_list)
        if cached is not None:
            return cached

    obs_ds = xr.open_dataset(nc_path)

    if os.path.exists(fit_path):
        fit_ds = xr.open_dataset(fit_path)
    else:
        _ensure_parent_dir(fit_path)
        train_ds = obs_ds.sel(time=slice(train_period[0], train_period[1]))
        xro_fit_model = XRO(ncycle=12, ac_order=2)
        fit_ds = xro_fit_model.fit_matrix(train_ds, maskb=["IOD"], maskNT=["T2", "TH"])
        fit_ds.to_netcdf(fit_path)

    xro_model = XRO(ncycle=12, ac_order=2)
    fcst = xro_model.reforecast(
        fit_ds=fit_ds,
        init_ds=obs_ds,
        n_month=max(lead_list) + 1,
        ncopy=1,
        noise_type="zero",
    )
    verify_period = slice(test_period[0], test_period[1])
    rmse = calc_forecast_skill(
        fcst,
        obs_ds,
        metric="rmse",
        is_mv3=True,
        by_month=False,
        verify_periods=verify_period,
    )
    acc = calc_forecast_skill(
        fcst,
        obs_ds,
        metric="acc",
        is_mv3=True,
        by_month=False,
        verify_periods=verify_period,
    )

    rmse_vals = [float(rmse["Nino34"].sel(lead=lead).values) for lead in lead_list]
    acc_vals = [float(acc["Nino34"].sel(lead=lead).values) for lead in lead_list]

    summary = {
        "label": label,
        "nc_path": nc_path,
        "train_period": list(train_period),
        "test_period": list(test_period),
        "fit_path": fit_path,
        "summary_path": summary_path,
        "csv_path": csv_path,
        "leads": list(lead_list),
        "rmse_by_lead": rmse_vals,
        "acc_by_lead": acc_vals,
        "mean_rmse_test_leads_0_21": float(np.nanmean([rmse for lead, rmse in zip(lead_list, rmse_vals) if lead in HISTORICAL_LEADS])),
        "mean_acc_test_leads_0_21": float(np.nanmean([acc for lead, acc in zip(lead_list, acc_vals) if lead in HISTORICAL_LEADS])),
        "mean_rmse_test_leads_1_20": float(np.nanmean([rmse for lead, rmse in zip(lead_list, rmse_vals) if lead in DIAGNOSTIC_LEADS])),
        "mean_acc_test_leads_1_20": float(np.nanmean([acc for lead, acc in zip(lead_list, acc_vals) if lead in DIAGNOSTIC_LEADS])),
        "note": (
            "Computed with the repo's current local calc_forecast_skill path so the XRO "
            "curve is directly comparable to the regenerated memory-model plots."
        ),
    }

    _save_curve_csv(csv_path, lead_list, rmse_vals, acc_vals)
    _ensure_parent_dir(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
