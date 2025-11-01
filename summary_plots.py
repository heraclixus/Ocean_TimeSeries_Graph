import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _is_graph(label: str) -> bool:
    s = label.lower()
    return ('graph' in s)


def _is_extra(label: str) -> bool:
    s = label.lower()
    return ('_sim' in s) or ('extra' in s)


def _strip_extra_suffix(label: str) -> str:
    # Remove any trailing _sim... or _extra_sim tokens
    s = label
    if '_sim' in s:
        s = s[: s.index('_sim')]
    if 'extra_sim' in s:
        s = s.replace('_extra_sim', '')
    return s


def _compute_rank_df(metric_map: Dict[str, any], higher_is_better: bool = True) -> pd.DataFrame:
    model_names = list(metric_map.keys())
    if len(model_names) == 0:
        return pd.DataFrame()
    # common lead set
    lead_sets = [set(metric_map[m]['lead'].values.tolist()) for m in model_names]
    common = sorted(list(set.intersection(*lead_sets)))
    rows = []
    for L in common:
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
        rows.append((L, ranks))
    if not rows:
        return pd.DataFrame()
    leads_final = [L for (L, _) in rows]
    rank_mat = np.stack([r for (_, r) in rows], axis=0)
    df = pd.DataFrame(rank_mat, index=leads_final, columns=model_names)
    df.index.name = 'lead'
    return df


def _plot_overlay(metric_map: Dict[str, any], title: str, ylabel: str, out_path: str):
    if len(metric_map) == 0:
        return
    plt.figure(figsize=(9, 4))
    for label, da in metric_map.items():
        try:
            plt.plot(da['lead'].values, da.values, label=label)
        except Exception:
            continue
    plt.xlabel('Lead (months)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_bar(labels: list, values: list, title: str, ylabel: str, out_path: str):
    if len(labels) == 0:
        return
    width = max(8, 0.6 * len(labels))
    plt.figure(figsize=(width, 4))
    x = np.arange(len(labels))
    plt.bar(x, values, alpha=0.85)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_all_summaries(acc_map: Dict[str, any], rmse_map: Dict[str, any], out_dir: str = 'results/summary', fig_tag: str = ''):
    """Create summary ablation plots from metric maps keyed by model label -> DataArray (lead).

    Ablations:
      1) Extra training data effect (paired comparison by base label).
      2) Graph-only model comparison (ACC and RMSE overlays).
      3) Graph vs Non-graph category comparison (mean curves).
      4) Average ranks of all models (ACC higher-is-better, RMSE lower-is-better).
    """
    _ensure_dir(out_dir)

    # 1) Extra data effect (pair baseline vs extra for any model where both exist)
    def build_pairs(metric_map: Dict[str, any]):
        base_to_variants = {}
        for label in metric_map.keys():
            base = _strip_extra_suffix(label)
            base_to_variants.setdefault(base, []).append(label)
        pairs = []
        for base, labels in base_to_variants.items():
            base_lab = None
            extra_lab = None
            for lb in labels:
                if _is_extra(lb):
                    extra_lab = lb
                else:
                    base_lab = lb
            if base_lab is not None and extra_lab is not None:
                pairs.append((base_lab, extra_lab))
        return pairs

    acc_pairs = build_pairs(acc_map)
    rmse_pairs = build_pairs(rmse_map)
    # Plot delta bars (average across leads)
    if acc_pairs:
        labels = []
        deltas = []
        for base_lab, extra_lab in acc_pairs:
            a0 = float(np.nanmean(acc_map[base_lab].values))
            a1 = float(np.nanmean(acc_map[extra_lab].values))
            labels.append(_strip_extra_suffix(extra_lab))
            deltas.append(a1 - a0)  # + means extra improved ACC
        _plot_bar(labels, deltas, 'ACC improvement with extra data (avg over leads)', 'ΔACC (extra - base)', os.path.join(out_dir, f'extra_acc_delta_bar{fig_tag}.png'))
    if rmse_pairs:
        labels = []
        deltas = []
        for base_lab, extra_lab in rmse_pairs:
            r0 = float(np.nanmean(rmse_map[base_lab].values))
            r1 = float(np.nanmean(rmse_map[extra_lab].values))
            labels.append(_strip_extra_suffix(extra_lab))
            deltas.append(r0 - r1)  # + means extra reduced RMSE
        _plot_bar(labels, deltas, 'RMSE reduction with extra data (avg over leads)', 'ΔRMSE (base - extra)', os.path.join(out_dir, f'extra_rmse_delta_bar{fig_tag}.png'))

    # 2) Graph-only comparison overlays
    acc_graph = {k: v for k, v in acc_map.items() if _is_graph(k)}
    rmse_graph = {k: v for k, v in rmse_map.items() if _is_graph(k)}
    _plot_overlay(acc_graph, 'Graph models ACC vs lead', 'Correlation', os.path.join(out_dir, f'graphs_acc_overlay{fig_tag}.png'))
    _plot_overlay(rmse_graph, 'Graph models RMSE vs lead', 'RMSE (℃)', os.path.join(out_dir, f'graphs_rmse_overlay{fig_tag}.png'))

    # 3) Graph vs Non-graph mean curves
    acc_non = {k: v for k, v in acc_map.items() if not _is_graph(k)}
    rmse_non = {k: v for k, v in rmse_map.items() if not _is_graph(k)}
    def mean_curve(metric_map: Dict[str, any]):
        if not metric_map:
            return None
        leads = list(metric_map.values())[0]['lead'].values
        arr = np.stack([m.values for m in metric_map.values()], axis=0)
        mu = np.nanmean(arr, axis=0)
        return leads, mu
    acc_g = mean_curve(acc_graph); acc_n = mean_curve(acc_non)
    if acc_g is not None or acc_n is not None:
        plt.figure(figsize=(8, 4))
        if acc_g is not None:
            plt.plot(acc_g[0], acc_g[1], label='Graph mean')
        if acc_n is not None:
            plt.plot(acc_n[0], acc_n[1], label='Non-graph mean')
        plt.xlabel('Lead (months)'); plt.ylabel('Correlation'); plt.title('ACC: Graph vs Non-graph mean'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'graph_vs_nongraph_acc_mean{fig_tag}.png'), dpi=300)
        plt.close()
    rmse_g = mean_curve(rmse_graph); rmse_n = mean_curve(rmse_non)
    if rmse_g is not None or rmse_n is not None:
        plt.figure(figsize=(8, 4))
        if rmse_g is not None:
            plt.plot(rmse_g[0], rmse_g[1], label='Graph mean')
        if rmse_n is not None:
            plt.plot(rmse_n[0], rmse_n[1], label='Non-graph mean')
        plt.xlabel('Lead (months)'); plt.ylabel('RMSE (℃)'); plt.title('RMSE: Graph vs Non-graph mean'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'graph_vs_nongraph_rmse_mean{fig_tag}.png'), dpi=300)
        plt.close()

    # 4) Average ranks across all models
    acc_rank = _compute_rank_df(acc_map, higher_is_better=True)
    rmse_rank = _compute_rank_df(rmse_map, higher_is_better=False)
    if not acc_rank.empty:
        acc_overall = acc_rank.mean(axis=0).sort_values()
        acc_overall.to_frame('avg_rank').to_csv(os.path.join(out_dir, 'overall_rank_acc.csv'))
        _plot_bar(list(acc_overall.index), list(acc_overall.values), 'Overall average rank by ACC', 'Avg rank (1=best)', os.path.join(out_dir, f'overall_rank_acc_bar{fig_tag}.png'))
    if not rmse_rank.empty:
        rmse_overall = rmse_rank.mean(axis=0).sort_values()
        rmse_overall.to_frame('avg_rank').to_csv(os.path.join(out_dir, 'overall_rank_rmse.csv'))
        _plot_bar(list(rmse_overall.index), list(rmse_overall.values), 'Overall average rank by RMSE', 'Avg rank (1=best)', os.path.join(out_dir, f'overall_rank_rmse_bar{fig_tag}.png'))


