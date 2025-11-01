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
from utils.xro_utils import calc_forecast_skill, nxro_reforecast
from XRO.core import XRO
from nxro.models import NXROGraphModel, NXROGraphPyGModel
from graph_construction import get_or_build_xro_graph, get_or_build_stat_knn_graph


def main():
    parser = argparse.ArgumentParser(description='Compare all graph-based NXRO variants')
    parser.add_argument('--test', action='store_true', help='Use only test-suffixed checkpoints (*_best_test*.pt)')
    parser.add_argument('--select_metric', choices=['rmse', 'acc', 'combined'], default='rmse',
                        help='Metric to select best checkpoint per config')
    parser.add_argument('--data_filter', choices=['all', 'base', 'extra'], default='all',
                        help='Filter graph models by whether trained with extra sim data')
    args = parser.parse_args()

    os.makedirs('results/graph_comparison', exist_ok=True)
    
    # Data
    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    train_ds = obs_ds.sel(time=slice('1979-01', '2022-12'))

    # Evaluation period
    eval_start = '2023-01' if args.test else '1979-01'
    eval_end = None if args.test else '2022-12'

    # ------- Baseline XRO -------
    XROac2 = XRO(ncycle=12, ac_order=2)
    XROac2_fit = XROac2.fit_matrix(train_ds, maskb=['IOD'], maskNT=['T2', 'TH'])
    XROac2_fcst = XROac2.reforecast(fit_ds=XROac2_fit, init_ds=obs_ds, n_month=21, ncopy=1, noise_type='zero')

    def _path_is_extra(p: str) -> bool:
        return ('_sim' in os.path.basename(p).lower()) or ('extra' in os.path.basename(p).lower())

    def _config_key_from_path(p: str) -> str:
        base = os.path.basename(p).replace('.pt', '')
        i = base.find('_best')
        return base[:i] if i >= 0 else base

    def _score_of_fcst(fcst: xr.Dataset) -> float:
        rmse_da = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice(eval_start, eval_end))
        acc_da = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice(eval_start, eval_end))
        try:
            mean_rmse = float(np.nanmean(rmse_da['Nino34'].values))
            mean_acc = float(np.nanmean(acc_da['Nino34'].values))
        except Exception:
            mean_rmse = float(np.nanmean(list(rmse_da.data_vars.values())[0].values))
            mean_acc = float(np.nanmean(list(acc_da.data_vars.values())[0].values))
        if args.select_metric == 'rmse':
            return mean_rmse
        if args.select_metric == 'acc':
            return -mean_acc
        return mean_rmse - mean_acc

    def _select_best_by_config_graph(paths):
        # Pre-filter by extra/base
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
            best_score = float('inf')
            best_path = None
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
                        best_score = score
                        best_path = p
                except Exception as e:
                    print(f"Warning: failed to load {p}: {e}")
                    continue
            if best_path is not None:
                selected.append(best_path)
        return selected

    def _select_best_by_config_pyg(paths):
        # Pre-filter by extra/base
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
            best_score = float('inf')
            best_path = None
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
                        best_score = score
                        best_path = p
                except Exception as e:
                    print(f"Warning: failed to load {p}: {e}")
                    continue
            if best_path is not None:
                selected.append(best_path)
        return selected

    # Collect graph forecasts
    graph_fcsts = {}
    graph_models = {}

    # Load NXRO-Graph variants
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
            sd_keys = list(ckpt['state_dict'].keys())
            if any(k.startswith('conv') for k in sd_keys) or ('edge_index' in sd_keys):
                continue
            base = os.path.basename(p)
            use_fixed = ('_fixed_' in base) or ('_learned_' not in base)
            nxro_gr_model = NXROGraphModel(n_vars=len(gr_var_order), k_max=2, use_fixed_graph=use_fixed)
            nxro_gr_model.load_state_dict(ckpt['state_dict'])
            label = base.replace('.pt', '').replace('nxro_', '').replace('_best_test', '').replace('_best', '')
            graph_fcsts[label] = nxro_reforecast(nxro_gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device='cpu')
            graph_models[label] = (nxro_gr_model, gr_var_order)
            print(f"Loaded NXRO-Graph: {label}")

    # Load NXRO-GraphPyG variants
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
        edge_index = _edge_index_from_adj(A, top_k=top_k)
        pyg_model = NXROGraphPyGModel(n_vars=len(vo), k_max=2, edge_index=edge_index, hidden=16, dropout=0.1, use_gat=use_gat)
        pyg_model.load_state_dict(ckpt['state_dict'])
        label = base.replace('nxro_','').replace('_best_test', '').replace('_best','')
        graph_fcsts[label] = nxro_reforecast(pyg_model, init_ds=obs_ds, n_month=21, var_order=vo, device='cpu')
        graph_models[label] = (pyg_model, vo)
        print(f"Loaded NXRO-GraphPyG: {label}")

    if not graph_fcsts:
        print("No graph models found!")
        return

    # Compute skills
    acc_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='acc', is_mv3=True,
                                     by_month=False, verify_periods=slice(eval_start, eval_end))
    rmse_XROac2 = calc_forecast_skill(XROac2_fcst, obs_ds, metric='rmse', is_mv3=True,
                                      by_month=False, verify_periods=slice(eval_start, eval_end))
    
    acc_graph = {}
    rmse_graph = {}
    for label, fcst in graph_fcsts.items():
        acc_graph[label] = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                              by_month=False, verify_periods=slice(eval_start, eval_end))
        rmse_graph[label] = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                               by_month=False, verify_periods=slice(eval_start, eval_end))

    sel_var = 'Nino34'
    fig_tag = '_test' if args.test else ''
    if args.data_filter != 'all':
        fig_tag += f'_{args.data_filter}'

    # ------- ACC Skill Curve -------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    acc_XROac2[sel_var].plot(ax=ax, label='XRO (baseline)', c='orangered', lw=3, ls='--')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(acc_graph)))
    for idx, (label, ds) in enumerate(acc_graph.items()):
        ds[sel_var].plot(ax=ax, label=label, c=colors[idx], lw=2)
    
    ax.set_ylabel('Anomaly Correlation Coefficient', fontsize=12)
    ax.set_xlabel('Forecast lead (months)', fontsize=12)
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0.2, 1.])
    ax.set_xlim([1., 21])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_title(f'ACC Comparison: All Graph Models vs XRO ({eval_start} onwards)', fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/graph_comparison/all_graph_acc{fig_tag}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/graph_comparison/all_graph_acc{fig_tag}.png")

    # ------- RMSE Skill Curve -------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rmse_XROac2[sel_var].plot(ax=ax, label='XRO (baseline)', c='orangered', lw=3, ls='--')
    
    for idx, (label, ds) in enumerate(rmse_graph.items()):
        ds[sel_var].plot(ax=ax, label=label, c=colors[idx], lw=2)
    
    ax.set_ylabel('RMSE (℃)', fontsize=12)
    ax.set_xlabel('Forecast lead (months)', fontsize=12)
    ax.set_xticks(np.arange(1, 24, step=2))
    ax.set_ylim([0., 1.2])
    ax.set_xlim([1., 21])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_title(f'RMSE Comparison: All Graph Models vs XRO ({eval_start} onwards)', fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/graph_comparison/all_graph_rmse{fig_tag}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: results/graph_comparison/all_graph_rmse{fig_tag}.png")

    # ------- Compute Rankings -------
    def compute_rank_stats(metric_map, higher_is_better=True):
        model_names = list(metric_map.keys())
        lead_sets = [set(metric_map[m]['lead'].values.tolist()) for m in model_names]
        common_leads = sorted(list(set.intersection(*lead_sets)))
        
        ranks_by_lead = []
        for L in common_leads:
            vals = []
            for m in model_names:
                v = metric_map[m].sel(lead=L).values
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            vals = np.array(vals, dtype=float)
            if np.isnan(vals).any():
                continue
            order = np.argsort(-vals) if higher_is_better else np.argsort(vals)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            ranks_by_lead.append(ranks)
        
        if ranks_by_lead:
            avg_ranks = np.mean(ranks_by_lead, axis=0)
            return dict(zip(model_names, avg_ranks))
        else:
            return {}

    # Build metric maps
    acc_map = {'XRO': acc_XROac2[sel_var]}
    for label, ds in acc_graph.items():
        acc_map[label] = ds[sel_var]
    
    rmse_map = {'XRO': rmse_XROac2[sel_var]}
    for label, ds in rmse_graph.items():
        rmse_map[label] = ds[sel_var]
    
    acc_ranks = compute_rank_stats(acc_map, higher_is_better=True)
    rmse_ranks = compute_rank_stats(rmse_map, higher_is_better=False)

    # ------- ACC Rank Bar Plot -------
    if acc_ranks:
        sorted_acc = dict(sorted(acc_ranks.items(), key=lambda x: x[1]))
        fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.6*len(sorted_acc)), 5))
        labels = list(sorted_acc.keys())
        vals = list(sorted_acc.values())
        colors_bar = ['orangered' if l == 'XRO' else 'steelblue' for l in labels]
        ax.bar(np.arange(len(labels)), vals, color=colors_bar, alpha=0.8, edgecolor='black')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Average Rank (lower is better)', fontsize=12)
        ax.set_title(f'Average ACC Rank: Graph Models vs XRO ({eval_start} onwards)', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/graph_comparison/all_graph_acc_rank_bar{fig_tag}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: results/graph_comparison/all_graph_acc_rank_bar{fig_tag}.png")
        
        # Save to CSV
        pd.Series(sorted_acc).to_frame('avg_rank').to_csv(f'results/graph_comparison/all_graph_acc_ranks{fig_tag}.csv')
        print(f"Saved: results/graph_comparison/all_graph_acc_ranks{fig_tag}.csv")

    # ------- RMSE Rank Bar Plot -------
    if rmse_ranks:
        sorted_rmse = dict(sorted(rmse_ranks.items(), key=lambda x: x[1]))
        fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.6*len(sorted_rmse)), 5))
        labels = list(sorted_rmse.keys())
        vals = list(sorted_rmse.values())
        colors_bar = ['orangered' if l == 'XRO' else 'steelblue' for l in labels]
        ax.bar(np.arange(len(labels)), vals, color=colors_bar, alpha=0.8, edgecolor='black')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Average Rank (lower is better)', fontsize=12)
        ax.set_title(f'Average RMSE Rank: Graph Models vs XRO ({eval_start} onwards)', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/graph_comparison/all_graph_rmse_rank_bar{fig_tag}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: results/graph_comparison/all_graph_rmse_rank_bar{fig_tag}.png")
        
        # Save to CSV
        pd.Series(sorted_rmse).to_frame('avg_rank').to_csv(f'results/graph_comparison/all_graph_rmse_ranks{fig_tag}.csv')
        print(f"Saved: results/graph_comparison/all_graph_rmse_ranks{fig_tag}.csv")

    print(f"\nFound {len(graph_fcsts)} graph model variants")
    print("=" * 60)


if __name__ == '__main__':
    main()

