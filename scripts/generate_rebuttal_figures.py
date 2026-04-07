"""Generate publication-quality figures for the KDD rebuttal."""
import glob
import re
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = 'tex/rebuttal/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'XRO': '#888888',
    'linear': '#2196F3',
    'attentive': '#E91E63',
    'graph_pyg': '#4CAF50',
    'res': '#FF9800',
    'pure_neural_ode': '#9C27B0',
    'pure_transformer': '#795548',
}
LABELS = {
    'XRO': 'XRO (baseline)',
    'linear': 'NXRO-Linear',
    'attentive': 'NXRO-Attentive',
    'graph_pyg': 'NXRO-GNN',
    'res': 'NXRO-MLP',
    'pure_neural_ode': 'Neural ODE',
    'pure_transformer': 'Transformer',
}


def extract_from_logs(pattern, min_jid, models, n_per_model):
    results = {}
    for f in sorted(glob.glob(pattern)):
        jid = int(re.search(r'_(\d+)\.out', f).group(1))
        if jid < min_jid:
            continue
        task = int(re.search(r'_(\d+)_\d+\.out', os.path.basename(f)).group(1))
        m_idx = task // n_per_model
        if m_idx >= len(models):
            continue
        model = models[m_idx]
        with open(f) as fh:
            text = fh.read()
        epoch_lines = re.findall(r'test RMSE: ([\d.]+)', text)
        if epoch_lines and 'Done' in text:
            results.setdefault(model, []).append(float(epoch_lines[-1]))
    return results


# =========================================================================
# Figure 1: Multi-seed bar chart with error bars
# =========================================================================
def fig1_multiseed_bar():
    models6 = ['linear', 'res', 'attentive', 'graph_pyg', 'pure_neural_ode', 'pure_transformer']
    ms = extract_from_logs('slurm/logs/multiseed_*_*.out', 52065, models6, 10)

    # Override res with mlp_best (h=4) results
    mlp_best = extract_from_logs('slurm/logs/mlp_best_*_*.out', 52370, ['res'], 1)
    if 'res' in mlp_best:
        ms['res'] = mlp_best['res']

    models_plot = ['attentive', 'graph_pyg', 'res', 'pure_transformer', 'pure_neural_ode']
    means = [np.mean(ms[m]) for m in models_plot]
    stds = [np.std(ms[m]) for m in models_plot]
    colors = [COLORS[m] for m in models_plot]
    labels = [LABELS[m] for m in models_plot]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models_plot))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black',
                  linewidth=0.5, width=0.6, zorder=3)
    ax.axhline(y=0.605, color=COLORS['XRO'], linestyle='--', linewidth=1.5,
               label='XRO baseline (0.605)', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Test RMSE (1-step, 10 seeds)')
    ax.set_title('Multi-Seed Validation: Train/Val/Test Split')
    ax.legend(loc='upper left')
    ax.set_ylim(0.48, 0.85)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    # Annotate values
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.008, f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.savefig(f'{OUT_DIR}/fig1_multiseed_barplot.png')
    plt.savefig(f'{OUT_DIR}/fig1_multiseed_barplot.pdf')
    plt.close()
    print(f'  Saved fig1_multiseed_barplot')


# =========================================================================
# Figure 2: Data scarcity curve
# =========================================================================
def fig2_data_scarcity():
    models4 = ['linear', 'res', 'attentive', 'graph_pyg']
    size_tags = ['10yr', '13yr', '16yr', '19yr', '23yr']
    size_years = [10, 13, 16, 19, 23]

    sc = {}
    for f in sorted(glob.glob('slurm/logs/data_scarcity_*_*.out')):
        jid = int(re.search(r'_(\d+)\.out', f).group(1))
        if jid < 52065:
            continue
        task = int(re.search(r'data_scarcity_(\d+)_', f).group(1))
        m_idx = task // 15
        rem = task % 15
        sz_idx = rem // 3
        if m_idx >= 4 or sz_idx >= 5:
            continue
        model = models4[m_idx]
        size = size_tags[sz_idx]
        with open(f) as fh:
            text = fh.read()
        epoch_lines = re.findall(r'test RMSE: ([\d.]+)', text)
        if epoch_lines and 'Done' in text:
            sc.setdefault((model, size), []).append(float(epoch_lines[-1]))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Override res (MLP) with h=4 scarcity results
    size_tags_sc = ['10yr', '13yr', '16yr', '19yr', '23yr']
    mlp4_sc = {}
    for f in sorted(glob.glob('slurm/logs/sc_mlp4_*_*.out')):
        jid = int(re.search(r'_(\d+)\.out', f).group(1))
        if jid < 52400: continue
        task = int(re.search(r'sc_mlp4_(\d+)_', f).group(1))
        sz_idx = task // 3
        if sz_idx >= 5: continue
        size = size_tags_sc[sz_idx]
        with open(f) as fh: text = fh.read()
        epoch_lines = re.findall(r'test RMSE: ([\d.]+)', text)
        if epoch_lines and 'Done' in text:
            mlp4_sc.setdefault(('res', size), []).append(float(epoch_lines[-1]))
    if mlp4_sc:
        for key, vals in mlp4_sc.items():
            sc[key] = vals

    for model in ['graph_pyg', 'attentive', 'res']:
        means = []
        stds = []
        for size in size_tags:
            v = sc.get((model, size), [])
            means.append(np.mean(v) if v else np.nan)
            stds.append(np.std(v) if v else 0)
        means = np.array(means)
        stds = np.array(stds)
        label = LABELS[model]
        ax.plot(size_years, means, 'o-', color=COLORS[model], label=label,
                linewidth=2, markersize=6, zorder=3)
        ax.fill_between(size_years, means - stds, means + stds,
                        color=COLORS[model], alpha=0.15, zorder=2)

    ax.axhline(y=0.605, color=COLORS['XRO'], linestyle='--', linewidth=1.5,
               label='XRO baseline (0.605)', zorder=1)

    ax.set_xlabel('Training data (years)')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Forecast Skill vs. Training Data Size')
    ax.set_xticks(size_years)
    ax.set_xticklabels(['10', '13', '16', '19', '23'])
    ax.legend(loc='upper right')
    ax.set_ylim(0.50, 0.82)
    ax.grid(alpha=0.3, zorder=0)

    plt.savefig(f'{OUT_DIR}/fig2_data_scarcity.png')
    plt.savefig(f'{OUT_DIR}/fig2_data_scarcity.pdf')
    plt.close()
    print(f'  Saved fig2_data_scarcity')


# =========================================================================
# Figure 3: Seasonal gate ablation — grouped bar by season
# =========================================================================
def fig3_seasonal_gate():
    csv_path = 'results_rebuttal_gate_ablation/seasonal_rmse_analysis.csv'
    if not os.path.exists(csv_path):
        print('  SKIP fig3: seasonal_rmse_analysis.csv not found')
        return

    df = pd.read_csv(csv_path)
    seasons = ['DJF', 'MAM', 'JJA', 'SON']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax_idx, model in enumerate(['attentive', 'graph_pyg']):
        ax = axes[ax_idx]
        sub = df[df['model'] == model]

        x = np.arange(len(seasons))
        width = 0.35

        for i, (cond, color, hatch) in enumerate([
            ('with_gate', COLORS[model], ''),
            ('no_gate', '#CCCCCC', '//'),
        ]):
            cond_data = sub[sub['condition'] == cond]
            means = []
            stds = []
            for season in seasons:
                vals = cond_data[cond_data['season'] == season]['rmse'].values
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            label = f'With gate' if cond == 'with_gate' else 'Without gate'
            bars = ax.bar(x + i * width - width / 2, means, width, yerr=stds,
                          capsize=3, color=color, edgecolor='black', linewidth=0.5,
                          hatch=hatch, label=label, zorder=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_title(LABELS[model])
        ax.set_ylabel('RMSE (avg over all leads)' if ax_idx == 0 else '')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_ylim(0.4, 1.35)

        # Highlight MAM
        ax.axvspan(0.5, 1.5, alpha=0.08, color='red', zorder=0)
        ax.text(1, ax.get_ylim()[1] * 0.96, 'Spring\nBarrier', ha='center',
                fontsize=8, color='red', fontstyle='italic')

    fig.suptitle('Seasonal Gate Ablation: RMSE by Initialization Season', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_seasonal_gate_ablation.png')
    plt.savefig(f'{OUT_DIR}/fig3_seasonal_gate_ablation.pdf')
    plt.close()
    print(f'  Saved fig3_seasonal_gate_ablation')


# =========================================================================
# Figure 4: Spring barrier detail — MAM per-lead RMSE
# =========================================================================
def fig4_spring_barrier_detail():
    csv_path = 'results_rebuttal_gate_ablation/seasonal_rmse_analysis.csv'
    if not os.path.exists(csv_path):
        print('  SKIP fig4: seasonal_rmse_analysis.csv not found')
        return

    df = pd.read_csv(csv_path)
    mam = df[df['season'] == 'MAM']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax_idx, model in enumerate(['attentive', 'graph_pyg']):
        ax = axes[ax_idx]
        sub = mam[mam['model'] == model]

        for cond, ls, lw, color in [
            ('with_gate', '-', 2.5, COLORS[model]),
            ('no_gate', '--', 2.0, '#888888'),
        ]:
            cond_data = sub[sub['condition'] == cond]
            pivot = cond_data.groupby('lead')['rmse'].agg(['mean', 'std'])
            leads = pivot.index.values
            means = pivot['mean'].values
            stds = pivot['std'].values
            label = f'With gate' if cond == 'with_gate' else 'Without gate'
            ax.plot(leads, means, ls, color=color, linewidth=lw, label=label,
                    marker='o', markersize=4, zorder=3)
            ax.fill_between(leads, means - stds, means + stds,
                            color=color, alpha=0.12, zorder=2)

        ax.set_xlabel('Forecast lead (months)')
        ax.set_ylabel('RMSE' if ax_idx == 0 else '')
        ax.set_title(f'{LABELS[model]} — MAM Initializations')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3, zorder=0)
        ax.set_xlim(1, 21)

    fig.suptitle('Spring Predictability Barrier: Effect of Seasonal Gate', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_spring_barrier_detail.png')
    plt.savefig(f'{OUT_DIR}/fig4_spring_barrier_detail.pdf')
    plt.close()
    print(f'  Saved fig4_spring_barrier_detail')


# =========================================================================
# Figure 5: Stochastic ablation — CRPS comparison
# =========================================================================
def fig5_stochastic_crps():
    stoch_crps = {}
    for model_dir in ['res', 'attentive', 'graphpyg']:
        dn = 'graph_pyg' if model_dir == 'graphpyg' else model_dir
        for noise in ['posthoc', 'stage2']:
            files = glob.glob(os.path.join(
                f'results_rebuttal_stochastic_ablation/{model_dir}',
                '**', f'*{noise}*seed*lead_metrics*'), recursive=True)
            crps_by_lead = {}
            for f in files:
                dfl = pd.read_csv(f)
                for _, row in dfl.iterrows():
                    if row['lead'] > 0:
                        crps_by_lead.setdefault(int(row['lead']), []).append(row['crps'])
            if crps_by_lead:
                stoch_crps[(dn, noise)] = crps_by_lead

    if not stoch_crps:
        print('  SKIP fig5: no CRPS data found')
        return

    # Load XRO baseline CRPS
    xro_crps_by_lead = {}
    xro_path = 'results_out_of_sample/xro_baseline/xro_stochastic_eval_lead_metrics.csv'
    if os.path.exists(xro_path):
        xro_df = pd.read_csv(xro_path)
        for _, row in xro_df.iterrows():
            if row['lead'] > 0:
                xro_crps_by_lead[int(row['lead'])] = row['crps']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    model_list = ['res', 'attentive', 'graph_pyg']

    for ax_idx, model in enumerate(model_list):
        ax = axes[ax_idx]

        # Plot XRO reference first (in each subplot)
        if xro_crps_by_lead:
            xro_leads = sorted(xro_crps_by_lead.keys())
            xro_vals = [xro_crps_by_lead[l] for l in xro_leads]
            ax.plot(xro_leads, xro_vals, '--', color=COLORS['XRO'], linewidth=1.8,
                    label='XRO (baseline)', zorder=2)

        for noise, ls, label in [('posthoc', ':', 'NXRO post-hoc AR(1)'), ('stage2', '-', 'NXRO Stage 2 (likelihood)')]:
            data = stoch_crps.get((model, noise), {})
            if not data:
                continue
            leads = sorted(data.keys())
            means = [np.mean(data[l]) for l in leads]
            stds = [np.std(data[l]) for l in leads]
            color = COLORS[model]
            ax.plot(leads, means, ls, color=color, linewidth=2, label=label,
                    marker='o', markersize=3, zorder=3)
            ax.fill_between(leads, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=color, alpha=0.1, zorder=2)

        ax.set_xlabel('Forecast lead (months)')
        ax.set_ylabel('CRPS' if ax_idx == 0 else '')
        ax.set_title(LABELS[model])
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3, zorder=0)
        ax.set_xlim(1, 21)

    fig.suptitle('Stochastic Forecast: CRPS by Lead (lower is better)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_stochastic_crps.png')
    plt.savefig(f'{OUT_DIR}/fig5_stochastic_crps.pdf')
    plt.close()
    print(f'  Saved fig5_stochastic_crps')


# =========================================================================
# Figure 6: Optimization decomposition — bar chart
# =========================================================================
def fig6_decomposition():
    models6 = ['linear', 'res', 'attentive', 'graph_pyg', 'pure_neural_ode', 'pure_transformer']
    ms = extract_from_logs('slurm/logs/multiseed_*_*.out', 52065, models6, 10)

    # Override res with mlp_best (h=4) results
    mlp_best = extract_from_logs('slurm/logs/mlp_best_*_*.out', 52370, ['res'], 1)
    if 'res' in mlp_best:
        ms['res'] = mlp_best['res']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Compare XRO vs NXRO variants vs pure neural — grouped bar
    model_groups = [
        ('XRO\n(baseline)', 0.605, 0, COLORS['XRO']),
        ('NXRO-Attentive', np.mean(ms['attentive']), np.std(ms['attentive']), COLORS['attentive']),
        ('NXRO-GNN', np.mean(ms['graph_pyg']), np.std(ms['graph_pyg']), COLORS['graph_pyg']),
        ('NXRO-MLP', np.mean(ms['res']), np.std(ms['res']), COLORS['res']),
        ('Transformer', np.mean(ms['pure_transformer']), np.std(ms['pure_transformer']), COLORS['pure_transformer']),
        ('Neural ODE', np.mean(ms['pure_neural_ode']), np.std(ms['pure_neural_ode']), COLORS['pure_neural_ode']),
    ]

    x = np.arange(len(model_groups))
    vals = [v for _, v, _, _ in model_groups]
    errs = [e for _, _, e, _ in model_groups]
    colors = [c for _, _, _, c in model_groups]
    labels = [l for l, _, _, _ in model_groups]

    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor='black',
                  linewidth=0.5, width=0.6, zorder=3)
    ax.axhline(y=0.605, color=COLORS['XRO'], linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
    ax.set_ylabel('Test RMSE (10 seeds)')
    ax.set_title('Validation-Selected Performance (Train/Val/Test Split)')
    ax.set_ylim(0.48, 0.85)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Annotate: NXRO hybrids beat XRO, pure neural doesn't
    ax.annotate('NXRO hybrid\nmodels', xy=(2, 0.50), fontsize=9, ha='center',
                color='green', fontstyle='italic')
    ax.annotate('Pure neural\n(no physics)', xy=(4.5, 0.50), fontsize=9, ha='center',
                color='red', fontstyle='italic')

    plt.savefig(f'{OUT_DIR}/fig6_decomposition.png')
    plt.savefig(f'{OUT_DIR}/fig6_decomposition.pdf')
    plt.close()
    print(f'  Saved fig6_decomposition')


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    print('Generating rebuttal figures...')
    fig1_multiseed_bar()
    fig2_data_scarcity()
    fig3_seasonal_gate()
    fig4_spring_barrier_detail()
    fig5_stochastic_crps()
    fig6_decomposition()
    print(f'\nAll figures saved to {OUT_DIR}/')
