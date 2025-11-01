import warnings
warnings.filterwarnings("ignore")
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import torch
import argparse
import glob
import pandas as pd
from utils.xro_utils import calc_forecast_skill, plot_forecast_plume, nxro_reforecast, evaluate_stochastic_ensemble
from nxro.stochastic import (
    compute_residuals_series,
    fit_seasonal_ar1_from_residuals,
    SeasonalAR1Noise,
    nxro_reforecast_stochastic,
)
from XRO.core import XRO
from nxro.train import train_nxro_linear, train_nxro_ro, train_nxro_rodiag, train_nxro_res, train_nxro_neural, train_nxro_bilinear, train_nxro_attentive, train_nxro_graph
from nxro.models import NXROGraphModel, NXROGraphPyGModel, NXRONeuralODEModel, NXROResidualMixModel
from graph_construction import get_or_build_xro_graph, get_or_build_stat_knn_graph
from utils.xro_utils import calc_forecast_skill, plot_forecast_plume, nxro_reforecast
from summary_plots import run_all_summaries


def main():
    parser = argparse.ArgumentParser(description='Compare XRO and NXRO variants')
    parser.add_argument('--test', action='store_true', help='Use only test-suffixed NXRO checkpoints (*_best_test_*.pt)')
    parser.add_argument('--stochastic', action='store_true', help='Compare stochastic ensemble versions')
    parser.add_argument('--members', type=int, default=100, help='Number of ensemble members when --stochastic')
    parser.add_argument('--ablation', action='store_true', help='Generate ablation summary plots')
    parser.add_argument('--plume', action='store_true', help='Generate forecast plume plots')
    parser.add_argument('--data_filter', choices=['all', 'base', 'extra'], default='base',
                        help='Filter NXRO models by whether trained with extra sim data')
    parser.add_argument('--graph_filter', choices=['all', 'graph', 'non'], default='all',
                        help='Filter NXRO models by graph vs non-graph')
    parser.add_argument('--select_metric', choices=['rmse', 'acc', 'combined'], default='rmse',
                        help='When deduping models per config and picking a single graph model, choose by this metric')
    parser.add_argument('--eval_all_datasets', action='store_true',
                        help='Evaluate on all available datasets (ORAS5, ERA5, GODAS, etc.) and generate comparison plots')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    
    # Helper to load all datasets
    def load_all_eval_datasets():
        """Load all available evaluation datasets."""
        datasets = {}
        datasets['ORAS5'] = xr.open_dataset('data/XRO_indices_oras5.nc')
        
        # Find all preprocessed datasets
        all_nc_files = glob.glob('data/XRO_indices_*_preproc.nc')
        for nc_file in all_nc_files:
            basename = os.path.basename(nc_file)
            dataset_name = basename.replace('XRO_indices_', '').replace('_preproc.nc', '').upper()
            if dataset_name and dataset_name != 'ORAS5':
                try:
                    datasets[dataset_name] = xr.open_dataset(nc_file)
                    print(f"  Loaded {dataset_name}: {nc_file}")
                except Exception as e:
                    print(f"  Warning: Could not load {nc_file}: {e}")
        return datasets
    
    # Data
    if args.eval_all_datasets:
        print("\nLoading all available datasets for evaluation...")
        all_eval_datasets = load_all_eval_datasets()
        print(f"✓ Loaded {len(all_eval_datasets)} datasets: {list(all_eval_datasets.keys())}\n")
        obs_ds = all_eval_datasets['ORAS5']  # Use ORAS5 as primary
    else:
        obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
        all_eval_datasets = None
    
    train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

    # Evaluation period
    eval_start = '2023-01' if args.test else '1979-01'
    eval_end = None if args.test else '2022-12'

    # ------- Baseline XRO (control, ac=2; linear; and ac=0) -------
    XROac2 = XRO(ncycle=12, ac_order=2)
    XROac0 = XRO(ncycle=12, ac_order=0)

    XROac2_fit = XROac2.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    XROac0_fit = XROac0.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    XROac2Lin_fit = XROac2.fit_matrix(train_ds, maskb=[], maskNT=[])

    # Deterministic reforecasts
    XROac2_fcst = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
    XROac0_fcst = XROac0.reforecast(fit_ds=XROac0_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')
    XROac2Lin_fcst = XROac2.reforecast(fit_ds=XROac2Lin_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')

    # Helper to find latest ckpt across multiple patterns
    def find_latest_multi(patterns, exclude_substr=None, require_extra=None) -> str:
        """
        Args:
            patterns: list of glob patterns to search
            exclude_substr: list of substrings to exclude from basename
            require_extra: if True, only keep files with extra data tag; if False, only keep files without; if None, don't filter
        """
        matches = []
        for patt in patterns:
            matches.extend(glob.glob(patt))
        if exclude_substr:
            matches = [m for m in matches if all(s not in os.path.basename(m) for s in exclude_substr)]
        if require_extra is not None:
            if require_extra:
                # Keep only files with extra data markers
                matches = [m for m in matches if _path_is_extra(m)]
            else:
                # Keep only files without extra data markers
                matches = [m for m in matches if not _path_is_extra(m)]
        if not matches:
            return ''
        matches.sort(key=os.path.getmtime)
        return matches[-1]
    
    def _path_is_extra(p: str) -> bool:
        return ('_sim' in os.path.basename(p).lower()) or ('extra' in os.path.basename(p).lower())

    # Collect NXRO forecasts dynamically
    nxro_fcsts = {}
    nxro_models = {}

    def _is_graph_label(lbl: str) -> bool:
        return 'graph' in lbl.lower()

    def _is_extra_label(lbl: str) -> bool:
        s = lbl.lower()
        return ('_sim' in s) or ('extra' in s)

    def _passes_filters(lbl: str) -> bool:
        if args.data_filter == 'extra' and not _is_extra_label(lbl):
            return False
        if args.data_filter == 'base' and _is_extra_label(lbl):
            return False
        if args.graph_filter == 'graph' and not _is_graph_label(lbl):
            return False
        if args.graph_filter == 'non' and _is_graph_label(lbl):
            return False
        return True

    def _config_key_from_path(p: str) -> str:
        base = os.path.basename(p).replace('.pt', '')
        i = base.find('_best')
        return base[:i] if i >= 0 else base

    def _select_latest_by_config(paths):
        by_key = {}
        for p in paths:
            key = _config_key_from_path(p)
            mt = os.path.getmtime(p)
            if key not in by_key or mt > by_key[key][1]:
                by_key[key] = (p, mt)
        return [v[0] for v in by_key.values()]

    def _score_of_fcst(fcst: xr.Dataset) -> float:
        # Lower is better
        rmse_da = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice(eval_start, eval_end))
        acc_da = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice(eval_start, eval_end))
        try:
            mean_rmse = float(np.nanmean(rmse_da['Nino34'].values))
            mean_acc = float(np.nanmean(acc_da['Nino34'].values))
        except Exception:
            # Fallback: use first variable
            mean_rmse = float(np.nanmean(list(rmse_da.data_vars.values())[0].values))
            mean_acc = float(np.nanmean(list(acc_da.data_vars.values())[0].values))
        if args.select_metric == 'rmse':
            return mean_rmse
        if args.select_metric == 'acc':
            return -mean_acc
        return mean_rmse - mean_acc

    def _select_best_by_config_graph(paths):
        # Optional pre-filter by extra/base
        if args.data_filter == 'base':
            paths = [p for p in paths if not _path_is_extra(p)]
        elif args.data_filter == 'extra':
            paths = [p for p in paths if _path_is_extra(p)]
        groups = {}
        for p in paths:
            key = _config_key_from_path(p)
            groups.setdefault(key, []).append(p)
        selected = []
        for key, plist in groups.items():
            best_score = float('inf'); best_path = None
            for p in plist:
                try:
                    ckpt = torch.load(p, map_location='cpu')
                    sd_keys = list(ckpt['state_dict'].keys())
                    if any(k.startswith('conv') for k in sd_keys) or ('edge_index' in sd_keys):
                        continue
                    gr_var_order = ckpt['var_order']
                    base = os.path.basename(p)
                    use_fixed = ('_fixed_' in base) or ('_learned_' not in base)
                    model = NXROGraphModel(n_vars=len(gr_var_order), k_max=2, use_fixed_graph=use_fixed)
                    model.load_state_dict(ckpt['state_dict'])
                    fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device='cpu')
                    score = _score_of_fcst(fcst)
                    if score < best_score:
                        best_score = score; best_path = p
                except Exception:
                    continue
            if best_path is not None:
                selected.append(best_path)
        return selected

    def _select_best_by_config_pyg(paths):
        # Optional pre-filter by extra/base
        if args.data_filter == 'base':
            paths = [p for p in paths if not _path_is_extra(p)]
        elif args.data_filter == 'extra':
            paths = [p for p in paths if _path_is_extra(p)]
        groups = {}
        for p in paths:
            key = _config_key_from_path(p)
            groups.setdefault(key, []).append(p)
        selected = []
        for key, plist in groups.items():
            best_score = float('inf'); best_path = None
            for p in plist:
                try:
                    ckpt = torch.load(p, map_location='cpu')
                    vo = ckpt['var_order']
                    base = os.path.basename(p).replace('.pt','')
                    toks = base.split('_')
                    use_gat = 'gat' in toks
                    k_tok = next((t for t in toks if t.startswith('k') and t[1:].isdigit()), 'k3')
                    top_k = int(k_tok[1:])
                    if 'stat' in toks:
                        i = toks.index('stat')
                        prior = toks[i+1] if i + 1 < len(toks) else 'pearson'
                    else:
                        prior = 'xro'
                    if prior == 'xro':
                        A, _ = get_or_build_xro_graph(nc_path='data/XRO_indices_oras5.nc', train_start='1979-01', train_end='2022-12', var_order=vo)
                    else:
                        A, _ = get_or_build_stat_knn_graph(data_path='data/XRO_indices_oras5_train.csv', train_start='1979-01', train_end='2022-12', var_order=vo, method=prior, top_k=top_k)
                    # edge_index builder inline
                    V = A.shape[0]
                    A2 = A.clone(); A2.fill_diagonal_(0.0); edges = []
                    for i in range(V):
                        vals, idx = torch.topk(A2[i], k=min(top_k, V - 1))
                        for j in idx.tolist():
                            if i != j and A2[i, j] > 0:
                                edges.append([i, j]); edges.append([j, i])
                    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.empty(2, 0, dtype=torch.long)
                    model = NXROGraphPyGModel(n_vars=len(vo), k_max=2, edge_index=edge_index, hidden=16, dropout=0.1, use_gat=use_gat)
                    model.load_state_dict(ckpt['state_dict'])
                    fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=vo, device='cpu')
                    score = _score_of_fcst(fcst)
                    if score < best_score:
                        best_score = score; best_path = p
                except Exception:
                    continue
            if best_path is not None:
                selected.append(best_path)
        return selected

    def _select_best_by_config_generic(paths, ctor):
        # Optional pre-filter by extra/base
        if args.data_filter == 'base':
            paths = [p for p in paths if not _path_is_extra(p)]
        elif args.data_filter == 'extra':
            paths = [p for p in paths if _path_is_extra(p)]
        groups = {}
        for p in paths:
            key = _config_key_from_path(p)
            groups.setdefault(key, []).append(p)
        selected = []
        for key, plist in groups.items():
            best_score = float('inf'); best_path = None
            for p in plist:
                try:
                    ckpt = torch.load(p, map_location='cpu')
                    vo = ckpt['var_order']
                    if ctor is NXROResidualMixModel:
                        sd_keys = ckpt['state_dict'].keys()
                        alpha_learnable = any('alpha_param' in k for k in sd_keys)
                        model = NXROResidualMixModel(n_vars=len(vo), k_max=2, hidden=64, alpha_init=0.1, alpha_learnable=alpha_learnable, alpha_max=0.5)
                    elif ctor is NXRONeuralODEModel:
                        model = NXRONeuralODEModel(n_vars=len(vo), k_max=2, hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only')
                    else:
                        continue
                    model.load_state_dict(ckpt['state_dict'])
                    fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=vo, device='cpu')
                    score = _score_of_fcst(fcst)
                    if score < best_score:
                        best_score = score; best_path = p
                except Exception:
                    continue
            if best_path is not None:
                selected.append(best_path)
        return selected

    # Determine require_extra based on data_filter
    require_extra = None if args.data_filter == 'all' else (True if args.data_filter == 'extra' else False)
    
    # ------- NXRO-Linear (load if exists, else train) -------
    linear_path = ''
    if args.test:
        linear_path = find_latest_multi([
            'results/linear/nxro_linear_best_test*.pt',
            'results/nxro_linear_best_test*.pt',
        ], require_extra=require_extra)
    else:
        linear_path = find_latest_multi([
            'results/linear/nxro_linear_best*.pt',
            'results/nxro_linear_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if linear_path and os.path.exists(linear_path):
        from nxro.models import NXROLinearModel
        ckpt = torch.load(linear_path, map_location='cpu')
        var_order = ckpt['var_order']
        nxro_model = NXROLinearModel(n_vars=len(var_order), k_max=2)
        nxro_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-Linear'
        if '_sim' in os.path.basename(linear_path) or 'extra' in os.path.basename(linear_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')
        nxro_models[lbl] = (nxro_model, var_order)
    elif not args.test:
        nxro_model, var_order, best_rmse, _ = train_nxro_linear(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_model.state_dict(), 'var_order': var_order}, 'results/nxro_linear_best.pt')
        nxro_fcsts['NXRO-Linear'] = nxro_reforecast(nxro_model, init_ds=obs_ds, n_month=21, var_order=var_order, device='cpu')
        nxro_models['NXRO-Linear'] = (nxro_model, var_order)

    # ------- NXRO-RO (load if exists, else train) -------
    ro_path = ''
    if args.test:
        ro_path = find_latest_multi([
            'results/ro/nxro_ro_best_test*.pt',
            'results/nxro_ro_best_test*.pt',
        ], require_extra=require_extra)
    else:
        ro_path = find_latest_multi([
            'results/ro/nxro_ro_best*.pt',
            'results/nxro_ro_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if ro_path and os.path.exists(ro_path):
        from nxro.models import NXROROModel
        ckpt = torch.load(ro_path, map_location='cpu')
        ro_var_order = ckpt['var_order']
        nxro_ro_model = NXROROModel(n_vars=len(ro_var_order), k_max=2)
        nxro_ro_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-RO'
        if '_sim' in os.path.basename(ro_path) or 'extra' in os.path.basename(ro_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device='cpu')
        nxro_models[lbl] = (nxro_ro_model, ro_var_order)
    elif not args.test:
        nxro_ro_model, ro_var_order, _, _ = train_nxro_ro(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_ro_model.state_dict(), 'var_order': ro_var_order}, 'results/nxro_ro_best.pt')
        nxro_fcsts['NXRO-RO'] = nxro_reforecast(nxro_ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device='cpu')
        nxro_models['NXRO-RO'] = (nxro_ro_model, ro_var_order)

    # ------- NXRO-RO+Diag (load if exists, else train) -------
    rodiag_path = ''
    if args.test:
        rodiag_path = find_latest_multi([
            'results/rodiag/nxro_rodiag_best_test*.pt',
            'results/nxro_rodiag_best_test*.pt',
        ], require_extra=require_extra)
    else:
        rodiag_path = find_latest_multi([
            'results/rodiag/nxro_rodiag_best*.pt',
            'results/nxro_rodiag_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if rodiag_path and os.path.exists(rodiag_path):
        from nxro.models import NXRORODiagModel
        ckpt = torch.load(rodiag_path, map_location='cpu')
        rd_var_order = ckpt['var_order']
        nxro_rd_model = NXRORODiagModel(n_vars=len(rd_var_order), k_max=2)
        nxro_rd_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-RO+Diag'
        if '_sim' in os.path.basename(rodiag_path) or 'extra' in os.path.basename(rodiag_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device='cpu')
        nxro_models[lbl] = (nxro_rd_model, rd_var_order)
    elif not args.test:
        nxro_rd_model, rd_var_order, _, _ = train_nxro_rodiag(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_rd_model.state_dict(), 'var_order': rd_var_order}, 'results/nxro_rodiag_best.pt')
        nxro_fcsts['NXRO-RO+Diag'] = nxro_reforecast(nxro_rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device='cpu')
        nxro_models['NXRO-RO+Diag'] = (nxro_rd_model, rd_var_order)

    # ------- NXRO-Res (load if exists, else train) -------
    res_path = ''
    if args.test:
        res_path = find_latest_multi([
            'results/res/nxro_res_best_test*.pt',
            'results/nxro_res_best_test*.pt',
        ], require_extra=require_extra)
    else:
        res_path = find_latest_multi([
            'results/res/nxro_res_best*.pt',
            'results/nxro_res_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if res_path and os.path.exists(res_path):
        from nxro.models import NXROResModel
        ckpt = torch.load(res_path, map_location='cpu')
        rs_var_order = ckpt['var_order']
        nxro_rs_model = NXROResModel(n_vars=len(rs_var_order), k_max=2)
        nxro_rs_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-Res'
        if '_sim' in os.path.basename(res_path) or 'extra' in os.path.basename(res_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device='cpu')
        nxro_models[lbl] = (nxro_rs_model, rs_var_order)
    elif not args.test:
        nxro_rs_model, rs_var_order, _, _ = train_nxro_res(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, device='cpu'
        )
        torch.save({'state_dict': nxro_rs_model.state_dict(), 'var_order': rs_var_order}, 'results/nxro_res_best.pt')
        nxro_fcsts['NXRO-Res'] = nxro_reforecast(nxro_rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device='cpu')
        nxro_models['NXRO-Res'] = (nxro_rs_model, rs_var_order)

    # ------- NXRO-NeuralODE (load if exists, else train) -------
    neural_path = ''
    if args.test:
        neural_path = find_latest_multi([
            'results/neural/nxro_neural_best_test*.pt',
            'results/nxro_neural_best_test*.pt',
        ], require_extra=require_extra)
    else:
        neural_path = find_latest_multi([
            'results/neural/nxro_neural_best*.pt',
            'results/nxro_neural_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if neural_path and os.path.exists(neural_path):
        from nxro.models import NXRONeuralODEModel
        ckpt = torch.load(neural_path, map_location='cpu')
        nn_var_order = ckpt['var_order']
        nxro_nn_model = NXRONeuralODEModel(n_vars=len(nn_var_order), k_max=2, hidden=64, depth=2, dropout=0.1, allow_cross=False)
        nxro_nn_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-NeuralODE'
        if '_sim' in os.path.basename(neural_path) or 'extra' in os.path.basename(neural_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device='cpu')
        nxro_models[lbl] = (nxro_nn_model, nn_var_order)
    elif not args.test:
        nxro_nn_model, nn_var_order, _, _ = train_nxro_neural(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, hidden=64, depth=2, dropout=0.1,
            allow_cross=False, mask_mode='th_only', device='cpu'
        )
        torch.save({'state_dict': nxro_nn_model.state_dict(), 'var_order': nn_var_order}, 'results/nxro_neural_best.pt')
        nxro_fcsts['NXRO-NeuralODE'] = nxro_reforecast(nxro_nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device='cpu')
        nxro_models['NXRO-NeuralODE'] = (nxro_nn_model, nn_var_order)

    # ------- NXRO-Bilinear (load if exists, else train) -------
    bl_path = ''
    if args.test:
        bl_path = find_latest_multi([
            'results/bilinear/nxro_bilinear_best_test*.pt',
            'results/nxro_bilinear_best_test*.pt',
        ], require_extra=require_extra)
    else:
        bl_path = find_latest_multi([
            'results/bilinear/nxro_bilinear_best*.pt',
            'results/nxro_bilinear_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if bl_path and os.path.exists(bl_path):
        from nxro.models import NXROBilinearModel
        ckpt = torch.load(bl_path, map_location='cpu')
        bl_var_order = ckpt['var_order']
        nxro_bl_model = NXROBilinearModel(n_vars=len(bl_var_order), k_max=2, n_channels=2, rank=2)
        nxro_bl_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-Bilinear'
        if '_sim' in os.path.basename(bl_path) or 'extra' in os.path.basename(bl_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device='cpu')
        nxro_models[lbl] = (nxro_bl_model, bl_var_order)
    elif not args.test:
        nxro_bl_model, bl_var_order, _, _ = train_nxro_bilinear(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, n_channels=2, rank=2, device='cpu'
        )
        torch.save({'state_dict': nxro_bl_model.state_dict(), 'var_order': bl_var_order}, 'results/nxro_bilinear_best.pt')
        nxro_fcsts['NXRO-Bilinear'] = nxro_reforecast(nxro_bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device='cpu')
        nxro_models['NXRO-Bilinear'] = (nxro_bl_model, bl_var_order)

    # ------- NXRO-Attentive (load if exists, else train) -------
    at_path = ''
    if args.test:
        at_path = find_latest_multi([
            'results/attentive/nxro_attentive_best_test*.pt',
            'results/nxro_attentive_best_test*.pt',
        ], require_extra=require_extra)
    else:
        at_path = find_latest_multi([
            'results/attentive/nxro_attentive_best*.pt',
            'results/nxro_attentive_best*.pt',
        ], exclude_substr=['_best_test'], require_extra=require_extra)
    if at_path and os.path.exists(at_path):
        from nxro.models import NXROAttentiveModel
        ckpt = torch.load(at_path, map_location='cpu')
        at_var_order = ckpt['var_order']
        nxro_at_model = NXROAttentiveModel(n_vars=len(at_var_order), k_max=2, d=32, dropout=0.1, mask_mode='th_only')
        nxro_at_model.load_state_dict(ckpt['state_dict'])
        lbl = 'NXRO-Attentive'
        if '_sim' in os.path.basename(at_path) or 'extra' in os.path.basename(at_path).lower():
            lbl += ' [extra]'
        nxro_fcsts[lbl] = nxro_reforecast(nxro_at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device='cpu')
        nxro_models[lbl] = (nxro_at_model, at_var_order)
    elif not args.test:
        nxro_at_model, at_var_order, _, _ = train_nxro_attentive(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, d=32, dropout=0.1, mask_mode='th_only', device='cpu'
        )
        torch.save({'state_dict': nxro_at_model.state_dict(), 'var_order': at_var_order}, 'results/nxro_attentive_best.pt')
        nxro_fcsts['NXRO-Attentive'] = nxro_reforecast(nxro_at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device='cpu')
        nxro_models['NXRO-Attentive'] = (nxro_at_model, at_var_order)

    # ------- NXRO-NeuralODE (PhysReg) and NXRO-ResidualMix (load if exists) -------
    # Neural PhysReg shares architecture with NXRONeuralODEModel (regularization applied only in training)
    for patt_list, ctor, label_name in [
        ((['results/neural_phys/nxro_neural_phys_best*.pt'] if not args.test else ['results/neural_phys/nxro_neural_phys_best_test*.pt']), NXRONeuralODEModel, 'NXRO-NeuralODE (PhysReg)'),
        ((['results/resmix/nxro_resmix_best*.pt'] if not args.test else ['results/resmix/nxro_resmix_best_test*.pt']), NXROResidualMixModel, 'NXRO-ResidualMix'),
    ]:
        matches = []
        for patt in patt_list + (['results/nxro_neural_phys_best*.pt'] if ctor is NXRONeuralODEModel and not args.test else []) + (['results/nxro_neural_phys_best_test*.pt'] if ctor is NXRONeuralODEModel and args.test else []) + (['results/nxro_resmix_best*.pt'] if ctor is NXROResidualMixModel and not args.test else []) + (['results/nxro_resmix_best_test*.pt'] if ctor is NXROResidualMixModel and args.test else []):
            matches.extend(glob.glob(patt))
        matches = _select_best_by_config_generic(matches, ctor)
        for path in matches:
            ckpt = torch.load(path, map_location='cpu')
            vo = ckpt['var_order']
            # Heuristic for ResidualMix alpha learnable flag
            if ctor is NXROResidualMixModel:
                sd_keys = ckpt['state_dict'].keys()
                alpha_learnable = any('alpha_param' in k for k in sd_keys)
                model = NXROResidualMixModel(n_vars=len(vo), k_max=2, hidden=64, alpha_init=0.1, alpha_learnable=alpha_learnable, alpha_max=0.5)
            else:
                model = ctor(n_vars=len(vo), k_max=2, hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only') if ctor is NXRONeuralODEModel else ctor
                if not isinstance(model, (NXRONeuralODEModel, NXROResidualMixModel)):
                    continue
            model.load_state_dict(ckpt['state_dict'])
            label = os.path.basename(path).replace('.pt','').replace('nxro_','').replace('_best_test_', ' (test) ').replace('_best','')
            # Use the descriptive label if simple base name
            if label.startswith('neural_phys'):
                label = f'{label_name}'
            elif label.startswith('resmix'):
                label = f'{label_name}'
            nxro_fcsts[label] = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=vo, device='cpu')
            nxro_models[label] = (model, vo)

    # ------- NXRO-Graph (load if exists, else train) -------
    # Support multiple graph variants: try to discover all matching checkpoints
    gr_paths = []
    if args.test:
        gr_paths = sorted(glob.glob('results/nxro_graph_*_best_test*.pt'))
        gr_paths += sorted(glob.glob('results/graph/*/nxro_graph_*_best_test*.pt'))
    else:
        gr_paths = sorted(glob.glob('results/nxro_graph_*_best.pt'))
        gr_paths += sorted(glob.glob('results/graph/*/nxro_graph_*_best.pt'))
    if gr_paths:
        gr_paths = _select_best_by_config_graph(gr_paths)
        for p in gr_paths:
            ckpt = torch.load(p, map_location='cpu')
            gr_var_order = ckpt['var_order']
            # Safety: skip PyG checkpoints accidentally matched
            sd_keys = list(ckpt['state_dict'].keys())
            if any(k.startswith('conv') for k in sd_keys) or ('edge_index' in sd_keys):
                continue
            # Decide fixed vs learned from filename
            base = os.path.basename(p)
            use_fixed = ('_fixed_' in base) or ('_learned_' not in base)
            nxro_gr_model = NXROGraphModel(n_vars=len(gr_var_order), k_max=2, use_fixed_graph=use_fixed)
            nxro_gr_model.load_state_dict(ckpt['state_dict'])
            label = base.replace('.pt', '').replace('nxro_', '').replace('_best_test_', ' (test) ')
            label = label.replace('_best', '')
            nxro_fcsts[label] = nxro_reforecast(nxro_gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device='cpu')
            nxro_models[label] = (nxro_gr_model, gr_var_order)
    elif not args.test:
        nxro_gr_model, gr_var_order, _, _ = train_nxro_graph(
            nc_path='data/XRO_indices_oras5.nc',
            train_start='1979-01', train_end='2022-12',
            test_start='2023-01', test_end=None,
            n_epochs=200, batch_size=128, lr=1e-3, k_max=2, use_fixed_graph=True, device='cpu'
        )
        torch.save({'state_dict': nxro_gr_model.state_dict(), 'var_order': gr_var_order}, 'results/nxro_graph_fixed_xro_best.pt')
        nxro_fcsts['graph_fixed_xro'] = nxro_reforecast(nxro_gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device='cpu')
        nxro_models['graph_fixed_xro'] = (nxro_gr_model, gr_var_order)

    # ------- NXRO-GraphPyG (load if exists) -------
    pyg_paths = []
    if args.test:
        pyg_paths = sorted(glob.glob('results/nxro_graphpyg_*_best_test*.pt'))
        pyg_paths += sorted(glob.glob('results/graphpyg/*/nxro_graphpyg_*_best_test*.pt'))
    else:
        pyg_paths = sorted(glob.glob('results/nxro_graphpyg_*_best.pt'))
        pyg_paths += sorted(glob.glob('results/graphpyg/*/nxro_graphpyg_*_best.pt'))
    pyg_paths = _select_best_by_config_pyg(pyg_paths)
    def _edge_index_from_adj(A: torch.Tensor, top_k: int) -> torch.Tensor:
        V = A.shape[0]
        A2 = A.clone()
        A2.fill_diagonal_(0.0)
        edges = []
        for i in range(V):
            vals, idx = torch.topk(A2[i], k=min(top_k, V - 1))
            for j in idx.tolist():
                if i != j and A2[i, j] > 0:
                    edges.append([i, j]); edges.append([j, i])
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty(2, 0, dtype=torch.long)
    for p in pyg_paths:
        ckpt = torch.load(p, map_location='cpu')
        vo = ckpt['var_order']
        base = os.path.basename(p).replace('.pt','')
        toks = base.split('_')
        # tokens: nxro graphpyg <gcn/gat> [stat <method>] kX ...
        use_gat = 'gat' in toks
        k_tok = next((t for t in toks if t.startswith('k') and t[1:].isdigit()), 'k3')
        top_k = int(k_tok[1:])
        if 'stat' in toks:
            i = toks.index('stat')
            prior = toks[i+1] if i + 1 < len(toks) else 'pearson'
        else:
            prior = 'xro'
        # Build edge_index
        if prior == 'xro':
            A, _ = get_or_build_xro_graph(nc_path='data/XRO_indices_oras5.nc', train_start='1979-01', train_end='2022-12', var_order=vo)
        else:
            A, _ = get_or_build_stat_knn_graph(data_path='data/XRO_indices_oras5_train.csv', train_start='1979-01', train_end='2022-12', var_order=vo, method=prior, top_k=top_k)
        edge_index = _edge_index_from_adj(A, top_k=top_k)
        pyg_model = NXROGraphPyGModel(n_vars=len(vo), k_max=2, edge_index=edge_index, hidden=16, dropout=0.1, use_gat=use_gat)
        pyg_model.load_state_dict(ckpt['state_dict'])
        label = base.replace('nxro_','').replace('_best_test_', ' (test) ').replace('_best','')
        nxro_fcsts[label] = nxro_reforecast(pyg_model, init_ds=obs_ds, n_month=21, var_order=vo, device='cpu')
        nxro_models[label] = (pyg_model, vo)

    # Keep only the single best Graph ODE model overall (across all graph variants)
    graph_labels_all = [lab for lab in list(nxro_fcsts.keys()) if _is_graph_label(lab)]
    graph_labels_considered = [lab for lab in graph_labels_all if _passes_filters(lab)]
    if graph_labels_considered:
        best_lab = None
        best_score = float('inf')
        for lab in graph_labels_considered:
            try:
                score = _score_of_fcst(nxro_fcsts[lab])
            except Exception:
                continue
            if score < best_score:
                best_score = score
                best_lab = lab
        if best_lab is not None:
            for lab in graph_labels_considered:
                if lab != best_lab:
                    nxro_fcsts.pop(lab, None)
                    nxro_models.pop(lab, None)

    # ------- Multi-dataset evaluation (if enabled) -------
    if args.eval_all_datasets and all_eval_datasets:
        print("\n" + "="*80)
        print("MULTI-DATASET EVALUATION")
        print("="*80)
        print(f"Evaluating all models on {len(all_eval_datasets)} datasets")
        print("="*80 + "\n")
        
        # Evaluate XRO baselines on all datasets
        multi_xro_results = {}
        for ds_name, ds in all_eval_datasets.items():
            print(f"Evaluating XRO baselines on {ds_name}...")
            multi_xro_results[ds_name] = {
                'XRO': {},
                'XRO_ac0': {},
                'Linear XRO': {},
            }
            for xro_label, xro_fcst in [('XRO', XROac2_fcst), ('XRO_ac0', XROac0_fcst), ('Linear XRO', XROac2Lin_fcst)]:
                acc = calc_forecast_skill(xro_fcst, ds, metric='acc', is_mv3=True,
                                         by_month=False, verify_periods=slice('1979-01', '2022-12'))
                rmse = calc_forecast_skill(xro_fcst, ds, metric='rmse', is_mv3=True,
                                          by_month=False, verify_periods=slice('1979-01', '2022-12'))
                multi_xro_results[ds_name][xro_label] = {'acc': acc, 'rmse': rmse}
                print(f"  {xro_label}: ACC={float(np.nanmean(acc['Nino34'].values)):.3f}, RMSE={float(np.nanmean(rmse['Nino34'].values)):.3f}")
        
        # Evaluate NXRO models on all datasets
        multi_nxro_results = {}
        for label, fcst in nxro_fcsts.items():
            if not _passes_filters(label):
                continue
            print(f"\nEvaluating {label} on all datasets...")
            multi_nxro_results[label] = {}
            for ds_name, ds in all_eval_datasets.items():
                try:
                    acc = calc_forecast_skill(fcst, ds, metric='acc', is_mv3=True,
                                            by_month=False, verify_periods=slice('1979-01', '2022-12'))
                    rmse = calc_forecast_skill(fcst, ds, metric='rmse', is_mv3=True,
                                             by_month=False, verify_periods=slice('1979-01', '2022-12'))
                    multi_nxro_results[label][ds_name] = {'acc': acc, 'rmse': rmse}
                    print(f"  {ds_name}: ACC={float(np.nanmean(acc['Nino34'].values)):.3f}, RMSE={float(np.nanmean(rmse['Nino34'].values)):.3f}")
                except Exception as e:
                    print(f"  {ds_name}: Failed - {e}")
        
        # Generate multi-dataset comparison plots
        print("\nGenerating multi-dataset comparison plots...")
        
        # Plot ACC across datasets for top models
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for ds_name in all_eval_datasets.keys():
            for xro_label in ['XRO', 'XRO_ac0', 'Linear XRO']:
                if xro_label in multi_xro_results.get(ds_name, {}):
                    multi_xro_results[ds_name][xro_label]['acc']['Nino34'].plot(
                        ax=ax, label=f'{xro_label} ({ds_name})', linestyle='--', alpha=0.7)
        
        for label in list(multi_nxro_results.keys())[:3]:  # Top 3 NXRO models
            for ds_name in all_eval_datasets.keys():
                if ds_name in multi_nxro_results[label]:
                    multi_nxro_results[label][ds_name]['acc']['Nino34'].plot(
                        ax=ax, label=f'{label} ({ds_name})', marker='o', markersize=3)
        
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title('ACC across multiple datasets', fontsize=12)
        ax.set_xlim([1., 21])
        ax.set_ylim([0.2, 1.])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/variants_acc_multi_dataset{fig_tag}.png', dpi=300)
        plt.close()
        print("  ✓ Saved ACC multi-dataset plot")
        
        # Plot RMSE across datasets
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for ds_name in all_eval_datasets.keys():
            for xro_label in ['XRO', 'XRO_ac0', 'Linear XRO']:
                if xro_label in multi_xro_results.get(ds_name, {}):
                    multi_xro_results[ds_name][xro_label]['rmse']['Nino34'].plot(
                        ax=ax, label=f'{xro_label} ({ds_name})', linestyle='--', alpha=0.7)
        
        for label in list(multi_nxro_results.keys())[:3]:  # Top 3 NXRO models
            for ds_name in all_eval_datasets.keys():
                if ds_name in multi_nxro_results[label]:
                    multi_nxro_results[label][ds_name]['rmse']['Nino34'].plot(
                        ax=ax, label=f'{label} ({ds_name})', marker='o', markersize=3)
        
        ax.set_ylabel('RMSE (℃)', fontsize=11)
        ax.set_xlabel('Forecast lead (months)', fontsize=11)
        ax.set_title('RMSE across multiple datasets', fontsize=12)
        ax.set_xlim([1., 21])
        ax.set_ylim([0., 1.])
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/variants_rmse_multi_dataset{fig_tag}.png', dpi=300)
        plt.close()
        print("  ✓ Saved RMSE multi-dataset plot")
        print()
    
    # ------- Skills (ACC and RMSE) for Nino34 on primary dataset (ORAS5) -------
    acc_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_XROac0 = calc_forecast_skill(XROac0_fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_XROac2Lin = calc_forecast_skill(XROac2Lin_fcst, obs_ds, metric='acc', is_mv3=True,
                                        by_month=False, verify_periods=slice('1979-01', '2022-12'))
    acc_nxro = {}
    for label, fcst in nxro_fcsts.items():
        acc_nxro[label] = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice('1979-01', '2022-12'))

    sel_var = 'Nino34'
    
    # Identify worst model by mean ACC and RMSE to drop from plots
    def _find_worst_model(acc_dict, rmse_dict):
        """Find model with worst combined performance (lowest ACC, highest RMSE)."""
        candidates = set(acc_dict.keys()) & set(rmse_dict.keys())
        candidates = {k for k in candidates if not k.startswith('XRO')}  # Don't drop baseline XRO variants
        if not candidates:
            return None
        scores = {}
        for label in candidates:
            try:
                mean_acc = float(np.nanmean(acc_dict[label].values))
                mean_rmse = float(np.nanmean(rmse_dict[label].values))
                # Combined score: lower ACC (bad) + higher RMSE (bad)
                scores[label] = -mean_acc + mean_rmse
            except:
                continue
        if scores:
            return max(scores, key=scores.get)
        return None
    
    # Compute RMSE first to identify worst model
    rmse_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_XROac0 = calc_forecast_skill(XROac0_fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_XROac2Lin = calc_forecast_skill(XROac2Lin_fcst, obs_ds, metric='rmse', is_mv3=True,
                                         by_month=False, verify_periods=slice('1979-01', '2022-12'))
    rmse_nxro = {}
    for label, fcst in nxro_fcsts.items():
        rmse_nxro[label] = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice('1979-01', '2022-12'))
    
    worst_model = _find_worst_model(acc_nxro, rmse_nxro)
    if worst_model:
        print(f"Dropping worst performing model from plots: {worst_model}")
    
    # Compute preliminary rankings to identify top 5 models
    def _compute_avg_ranks(acc_dict, rmse_dict):
        """Compute average rank for each model across ACC and RMSE."""
        all_models = set(acc_dict.keys()) | set(rmse_dict.keys())
        model_avg_ranks = {}
        for model in all_models:
            ranks = []
            if model in acc_dict:
                # Rank by ACC (higher is better)
                acc_vals = {m: float(np.nanmean(acc_dict[m].values)) for m in acc_dict.keys()}
                sorted_acc = sorted(acc_vals.items(), key=lambda x: -x[1])
                acc_rank = next((i+1 for i, (m, v) in enumerate(sorted_acc) if m == model), None)
                if acc_rank:
                    ranks.append(acc_rank)
            if model in rmse_dict:
                # Rank by RMSE (lower is better)
                rmse_vals = {m: float(np.nanmean(rmse_dict[m].values)) for m in rmse_dict.keys()}
                sorted_rmse = sorted(rmse_vals.items(), key=lambda x: x[1])
                rmse_rank = next((i+1 for i, (m, v) in enumerate(sorted_rmse) if m == model), None)
                if rmse_rank:
                    ranks.append(rmse_rank)
            if ranks:
                model_avg_ranks[model] = np.mean(ranks)
        return model_avg_ranks
    
    all_acc_for_rank = {'XRO': acc_XROac2[sel_var], 'XRO_ac0': acc_XROac0[sel_var], 'Linear XRO': acc_XROac2Lin[sel_var]}
    all_acc_for_rank.update({k: v[sel_var] for k, v in acc_nxro.items() if _passes_filters(k)})
    all_rmse_for_rank = {'XRO': rmse_XROac2[sel_var], 'XRO_ac0': rmse_XROac0[sel_var], 'Linear XRO': rmse_XROac2Lin[sel_var]}
    all_rmse_for_rank.update({k: v[sel_var] for k, v in rmse_nxro.items() if _passes_filters(k)})
    
    avg_ranks = _compute_avg_ranks(all_acc_for_rank, all_rmse_for_rank)
    top5_models = set(sorted(avg_ranks, key=avg_ranks.get)[:5])
    print(f"Top 5 models for skill plots: {sorted(top5_models, key=lambda x: avg_ranks[x])}")
    
    # Plot ACC (top 5 only)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    color_map = {
        'XRO': '#FF1744',              # Bright red (distinct primary)
        'XRO_ac0': '#2196F3',          # Bright blue
        'Linear XRO': '#00BCD4',       # Cyan
        'NXRO-Linear': '#4CAF50',      # Green
        'NXRO-RO': '#9C27B0',          # Purple
        'NXRO-RO+Diag': '#FF6F00',     # Deep orange (NOT red)
        'NXRO-Res': '#00897B',         # Teal
        'NXRO-NeuralODE': '#424242',   # Dark grey
        'NXRO-Bilinear': '#FFA726',    # Light orange
        'NXRO-Attentive': '#EC407A',   # Pink
        'NXRO-Graph': '#78909C',       # Blue grey
        'NXRO-ResidualMix': '#1A237E', # Indigo (NOT red)
        'NXRO-NeuralODE (PhysReg)': '#00695C',  # Dark teal/green
    }
    if 'XRO' in top5_models:
        acc_XROac2[sel_var].plot(ax=ax, label='XRO', c=color_map['XRO'], lw=2)
    if 'XRO_ac0' in top5_models:
        acc_XROac0[sel_var].plot(ax=ax, label='XRO$_{ac=0}$', c=color_map['XRO_ac0'], lw=2)
    if 'Linear XRO' in top5_models:
        acc_XROac2Lin[sel_var].plot(ax=ax, label='Linear XRO', c=color_map['Linear XRO'], ls='None', marker='.', ms=8)
    for label, ds in acc_nxro.items():
        if not _passes_filters(label):
            continue
        if label not in top5_models:
            continue
        base_lbl = label.split(' [')[0]
        c = color_map.get(base_lbl, None)
        if c is not None:
            ds[sel_var].plot(ax=ax, label=label, c=c, lw=2)
        else:
            ds[sel_var].plot(ax=ax, label=label, lw=2)
    ax.set_ylabel('Correlation')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0.2, 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend(fontsize=8, loc='lower right')
    fig_tag = ''
    if args.test:
        fig_tag += '_test'
    if args.data_filter != 'all':
        fig_tag += f'_{args.data_filter}'
    if args.graph_filter != 'all':
        fig_tag += f'_{args.graph_filter}'
    if args.stochastic:
        fig_tag += '_stoc'
    plt.savefig(f'results/variants_acc_skill{fig_tag}.png', dpi=300)
    plt.close()

    # Plot RMSE (top 5 only)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if 'XRO' in top5_models:
        rmse_XROac2[sel_var].plot(ax=ax, label='XRO', c=color_map['XRO'], lw=2)
    if 'XRO_ac0' in top5_models:
        rmse_XROac0[sel_var].plot(ax=ax, label='XRO$_{ac=0}$', c=color_map['XRO_ac0'], lw=2)
    if 'Linear XRO' in top5_models:
        rmse_XROac2Lin[sel_var].plot(ax=ax, label='Linear XRO', c=color_map['Linear XRO'], ls='None', marker='.', ms=8)
    for label, ds in rmse_nxro.items():
        if not _passes_filters(label):
            continue
        if label not in top5_models:
            continue
        base_lbl = label.split(' [')[0]
        c = color_map.get(base_lbl, None)
        if c is not None:
            ds[sel_var].plot(ax=ax, label=label, c=c, lw=2)
        else:
            ds[sel_var].plot(ax=ax, label=label, lw=2)
    ax.set_ylabel('RMSE (℃)')
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.])
    ax.set_xlim([1., 21])
    ax.set_xlabel('Forecast lead (months)')
    ax.legend(fontsize=8, loc='lower right')
    plt.savefig(f'results/variants_rmse_skill{fig_tag}.png', dpi=300)
    plt.close()

    # ------- Optional: stochastic ensemble comparison -------
    if args.stochastic:
        # XRO stochastic
        XROac2_fcst_stoc = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=args.members, noise_type='red')
        xro_eval = evaluate_stochastic_ensemble(XROac2_fcst_stoc, obs_ds, var='Nino34', out_prefix='results/variants_xro_stochastic_eval')
        xro_mean = XROac2_fcst_stoc.mean('member')
        acc_xro_m = calc_forecast_skill(xro_mean, obs_ds, metric='acc', is_mv3=True,
                                        by_month=False, verify_periods=slice('1979-01', '2022-12'))
        rmse_xro_m = calc_forecast_skill(xro_mean, obs_ds, metric='rmse', is_mv3=True,
                                         by_month=False, verify_periods=slice('1979-01', '2022-12'))

        # NXRO stochastic for each loaded model
        stoc_evals = {}
        acc_stoc = {}
        rmse_stoc = {}
        for label, (model, var_order) in nxro_models.items():
            if not _passes_filters(label):
                continue
            # Fit seasonal AR(1) noise from train residuals
            resid, months = compute_residuals_series(model, train_ds, var_order, device='cpu')
            a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
            a1 = torch.tensor(a1_np, dtype=torch.float32)
            sigma = torch.tensor(sigma_np, dtype=torch.float32)
            noise = SeasonalAR1Noise(a1, sigma)
            fcst_m = nxro_reforecast_stochastic(model, init_ds=obs_ds, n_month=21, var_order=var_order,
                                                noise_model=noise, n_members=args.members, device='cpu')
            stoc_evals[label] = evaluate_stochastic_ensemble(fcst_m, obs_ds, var='Nino34', out_prefix=f'results/variants_{label.lower().replace(" ", "_")}_stochastic_eval')
            mean_m = fcst_m.mean('member')
            acc_stoc[label] = calc_forecast_skill(mean_m, obs_ds, metric='acc', is_mv3=True,
                                                  by_month=False, verify_periods=slice('1979-01', '2022-12'))
            rmse_stoc[label] = calc_forecast_skill(mean_m, obs_ds, metric='rmse', is_mv3=True,
                                                   by_month=False, verify_periods=slice('1979-01', '2022-12'))

        # Plot ensemble-mean skill curves
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        acc_xro_m['Nino34'].plot(ax=ax, label='XRO (stoc mean)', c=color_map['XRO'], lw=2)
        for label, ds in acc_stoc.items():
            base_lbl = label.split(' [')[0]
            c = color_map.get(base_lbl, None)
            if c is not None:
                ds['Nino34'].plot(ax=ax, label=f'{label} (stoc mean)', c=c, lw=2)
            else:
                ds['Nino34'].plot(ax=ax, label=f'{label} (stoc mean)', lw=2)
        ax.set_ylabel('Correlation')
        ax.set_xticks(np.arange(1, 24, step=2))
        ax.set_ylim([0.2, 1.])
        ax.set_xlim([1., 21])
        ax.set_xlabel('Forecast lead (months)')
        ax.legend(fontsize=8, loc='lower right')
        plt.savefig(f'results/variants_stochastic_acc_skill{fig_tag}.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        rmse_xro_m['Nino34'].plot(ax=ax, label='XRO (stoc mean)', c=color_map['XRO'], lw=2)
        for label, ds in rmse_stoc.items():
            base_lbl = label.split(' [')[0]
            c = color_map.get(base_lbl, None)
            if c is not None:
                ds['Nino34'].plot(ax=ax, label=f'{label} (stoc mean)', c=c, lw=2)
            else:
                ds['Nino34'].plot(ax=ax, label=f'{label} (stoc mean)', lw=2)
        ax.set_ylabel('RMSE (℃)')
        ax.set_xticks(np.arange(1, 24, step=2))
        ax.set_ylim([0., 1.])
        ax.set_xlim([1., 21])
        ax.set_xlabel('Forecast lead (months)')
        ax.legend(fontsize=8, loc='lower right')
        plt.savefig(f'results/variants_stochastic_rmse_skill{fig_tag}.png', dpi=300)
        plt.close()

        # Multi-model spread vs RMSE comparison
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(xro_eval['lead'], xro_eval['spread'], label='XRO spread', c=color_map['XRO'], ls='-')
        ax.plot(xro_eval['lead'], xro_eval['rmse_mean'], label='XRO RMSE(mean)', c=color_map['XRO'], ls='--')
        for label, dse in stoc_evals.items():
            base_lbl = label.split(' [')[0]
            c = color_map.get(base_lbl, None)
            if c is None:
                c = None
            ax.plot(dse['lead'], dse['spread'], label=f'{label} spread', c=c, ls='-')
            ax.plot(dse['lead'], dse['rmse_mean'], label=f'{label} RMSE(mean)', c=c, ls='--')
        ax.set_xlabel('Lead (months)')
        ax.set_ylabel('℃')
        ax.set_title('Spread vs RMSE (ensemble mean)')
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'results/variants_stochastic_spread_rmse_compare{fig_tag}.png', dpi=300)
        plt.close()

        # CRPS bar comparison
        crps_vals = []
        crps_labels = []
        crps_vals.append(float(np.nanmean(xro_eval['crps'].values)))
        crps_labels.append('XRO')
        for label, dse in stoc_evals.items():
            crps_vals.append(float(np.nanmean(dse['crps'].values)))
            crps_labels.append(label)
        fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.6 * len(crps_labels)), 4))
        ax.bar(np.arange(len(crps_labels)), crps_vals, color='tab:blue', alpha=0.8)
        ax.set_xticks(np.arange(len(crps_labels)))
        ax.set_xticklabels(crps_labels, rotation=45, ha='right')
        ax.set_ylabel('Avg CRPS (lower is better)')
        ax.set_title('CRPS comparison (averaged over leads)')
        plt.tight_layout()
        plt.savefig(f'results/variants_stochastic_crps_bar{fig_tag}.png', dpi=300)
        plt.close()

        # Coverage 80% bar comparison
        def cov80(ds_eval):
            if 'interval' in ds_eval and 0.8 in ds_eval['interval'].values:
                idx = int(np.where(ds_eval['interval'].values == 0.8)[0][0])
                return float(np.nanmean(ds_eval['coverage'].isel(interval=idx).values))
            return np.nan
        cov_vals = [cov80(xro_eval)]
        cov_labels = ['XRO']
        for label, dse in stoc_evals.items():
            cov_vals.append(cov80(dse))
            cov_labels.append(label)
        fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.6 * len(cov_labels)), 4))
        ax.bar(np.arange(len(cov_labels)), cov_vals, color='tab:green', alpha=0.8)
        ax.set_xticks(np.arange(len(cov_labels)))
        ax.set_xticklabels(cov_labels, rotation=45, ha='right')
        ax.set_ylabel('Avg coverage @80% (target 0.8)')
        ax.set_title('Coverage calibration (80% interval)')
        ax.axhline(0.8, c='k', ls='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'results/variants_stochastic_coverage80_bar{fig_tag}.png', dpi=300)
        plt.close()

    # ------- Rank tables and heatmaps (per-lead ranking across models) -------
    def compute_rank_df(metric_map, higher_is_better=True):
        model_names = list(metric_map.keys())
        # Common leads across all models
        lead_sets = [set(metric_map[m]['lead'].values.tolist()) for m in model_names]
        common_leads = sorted(list(set.intersection(*lead_sets)))
        rows = []
        for L in common_leads:
            vals = []
            for m in model_names:
                v = metric_map[m].sel(lead=L).values
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            vals = np.array(vals, dtype=float)
            # Skip lead if any NaN present
            if np.isnan(vals).any():
                continue
            order = np.argsort(-vals) if higher_is_better else np.argsort(vals)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            rows.append((L, ranks))
        leads_final = [L for (L, _) in rows]
        rank_mat = np.stack([r for (_, r) in rows], axis=0) if rows else np.zeros((0, len(model_names)), dtype=int)
        df = pd.DataFrame(rank_mat, index=leads_final, columns=model_names)
        df.index.name = 'lead'
        return df

    # Build metric maps for selected variable
    acc_map = {
        'XRO': acc_XROac2[sel_var],
        'XRO_ac0': acc_XROac0[sel_var],
        'Linear XRO': acc_XROac2Lin[sel_var],
    }
    for label, ds in acc_nxro.items():
        if not _passes_filters(label):
            continue
        acc_map[label] = ds[sel_var]
    rmse_map = {
        'XRO': rmse_XROac2[sel_var],
        'XRO_ac0': rmse_XROac0[sel_var],
        'Linear XRO': rmse_XROac2Lin[sel_var],
    }
    for label, ds in rmse_nxro.items():
        if not _passes_filters(label):
            continue
        rmse_map[label] = ds[sel_var]

    acc_rank_df = compute_rank_df(acc_map, higher_is_better=True)
    rmse_rank_df = compute_rank_df(rmse_map, higher_is_better=False)
    acc_rank_df.to_csv(f'results/variants_rank_acc{fig_tag}.csv')
    rmse_rank_df.to_csv(f'results/variants_rank_rmse{fig_tag}.csv')

    def plot_rank_heatmap(df: pd.DataFrame, title: str, out_path: str):
        if df.empty:
            return
        fig_w = max(8, 1.0 * len(df.columns))
        fig_h = max(4, 0.4 * len(df.index))
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        im = ax.imshow(df.values, aspect='auto', origin='lower', cmap='viridis_r')
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index)
        ax.set_xlabel('Model')
        ax.set_ylabel('Lead (months)')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank (1=best)')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    plot_rank_heatmap(acc_rank_df, f'Model ranks by ACC ({sel_var})', f'results/variants_rank_acc_heatmap{fig_tag}.png')
    plot_rank_heatmap(rmse_rank_df, f'Model ranks by RMSE ({sel_var})', f'results/variants_rank_rmse_heatmap{fig_tag}.png')

    # Overall rank (average across leads)
    def compute_overall_rank(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.mean(axis=0).sort_values()

    def plot_rank_bar(series: pd.Series, title: str, out_path: str):
        if series.empty:
            return
        fig_w = max(8, 0.6 * len(series.index))
        fig_h = 4
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        ax.bar(np.arange(len(series.index)), series.values, color='tab:blue', alpha=0.8)
        ax.set_xticks(np.arange(len(series.index)))
        ax.set_xticklabels(series.index, rotation=45, ha='right')
        ax.set_ylabel('Average rank (lower is better)')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    acc_overall = compute_overall_rank(acc_rank_df)
    rmse_overall = compute_overall_rank(rmse_rank_df)
    acc_overall.to_frame('avg_rank').to_csv(f'results/variants_overall_rank_acc{fig_tag}.csv')
    rmse_overall.to_frame('avg_rank').to_csv(f'results/variants_overall_rank_rmse{fig_tag}.csv')
    plot_rank_bar(acc_overall, f'Overall average rank by ACC ({sel_var})', f'results/variants_overall_rank_acc_bar{fig_tag}.png')
    plot_rank_bar(rmse_overall, f'Overall average rank by RMSE ({sel_var})', f'results/variants_overall_rank_rmse_bar{fig_tag}.png')

    # Combined overall rank (mean of ACC and RMSE average ranks)
    if not acc_overall.empty and not rmse_overall.empty:
        common_models = acc_overall.index.intersection(rmse_overall.index)
        combined_overall = ((acc_overall[common_models] + rmse_overall[common_models]) / 2.0).sort_values()
        combined_overall.to_frame('avg_rank').to_csv(f'results/variants_overall_rank_combined{fig_tag}.csv')
        plot_rank_bar(combined_overall, f'Combined overall average rank (ACC + RMSE) ({sel_var})', f'results/variants_overall_rank_combined_bar{fig_tag}.png')

    # Plumes (optional, only if --plume flag is set)
    if args.plume:
        XROac2_fcst_stoc = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=100, noise_type='red')
        dates = ['1997-04', '1997-12', '2022-09', '2022-12', '2024-12']
        plot_forecast_plume(XROac2_fcst, XROac2_fcst_stoc, obs_ds, dates, fname_prefix='results/variants_plume', fig_suffix=fig_tag)
        print("Generated plume plots in results/")

    # Summary ablation plots (extra data effect, graph comparison, graph vs non-graph, overall ranks)
    if args.ablation:
        run_all_summaries(acc_map, rmse_map, out_dir='results/summary', fig_tag=fig_tag)


if __name__ == '__main__':
    main()


