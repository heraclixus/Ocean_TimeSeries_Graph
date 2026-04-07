"""Final rebuttal results summary across all experiments."""
import glob, re, os
import numpy as np
import pandas as pd

XRO = 0.605
paper_ref = {'res': 0.579, 'attentive': 0.554, 'graph_pyg': 0.561,
             'pure_neural_ode': 0.918, 'pure_transformer': 0.701}

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

# ============================================================================
print("=" * 80)
print("FINAL REBUTTAL RESULTS SUMMARY")
print("=" * 80)

# 1. MULTISEED
models6 = ['linear', 'res', 'attentive', 'graph_pyg', 'pure_neural_ode', 'pure_transformer']
ms = extract_from_logs('slurm/logs/multiseed_*_*.out', 52065, models6, 10)

print("\n## 1. Multi-Seed (6yr val: train 79-95, val 96-01, test 02-22)")
print(f"{'Model':20s} {'N':>3s} {'RMSE':>8s} {'±std':>7s} {'Paper':>7s} {'vs XRO':>8s}")
print("-" * 58)
for m in models6:
    if m not in ms:
        continue
    v = ms[m]; mn = np.mean(v); st = np.std(v)
    p = paper_ref.get(m); ps = f"{p:.3f}" if p else "N/A"
    vx = f"{(mn-XRO)/XRO*100:+.1f}%"
    print(f"{m:20s} {len(v):3d} {mn:8.4f} {st:7.4f} {ps:>7s} {vx:>8s}")

# 2. NARROW VAL
models5 = ['linear', 'res', 'attentive', 'graph_pyg', 'pure_transformer']
nv = extract_from_logs('slurm/logs/narrow_val_*_*.out', 52320, models5, 3)

print(f"\n## 2. Narrow Val (3yr val: train 79-98, val 99-01, test 02-22)")
ref6 = {'linear': 0.5596, 'res': 0.7270, 'attentive': 0.5550, 'graph_pyg': 0.5567, 'pure_transformer': 0.6763}
print(f"{'Model':20s} {'N':>3s} {'RMSE':>8s} {'±std':>7s} {'6yr-val':>8s} {'Paper':>7s} {'vs XRO':>8s}")
print("-" * 66)
for m in models5:
    if m not in nv:
        continue
    v = nv[m]; mn = np.mean(v); st = np.std(v)
    r6 = ref6.get(m, 0); p = paper_ref.get(m)
    ps = f"{p:.3f}" if p else "N/A"
    vx = f"{(mn-XRO)/XRO*100:+.1f}%"
    print(f"{m:20s} {len(v):3d} {mn:8.4f} {st:7.4f} {r6:8.4f} {ps:>7s} {vx:>8s}")

# 3. MLP SWEEP TOP 5
print("\n## 3. MLP Sweep Top 5 (under 6yr val)")
hidden_values = [16, 32, 64]
wd_values = [0.01, 0.005, 0.001]
rr_values = ['1e-3', '1e-4', '1e-5']
mlp_results = {}
for f in sorted(glob.glob('slurm/logs/mlp_sweep_*_*.out')):
    jid = int(re.search(r'_(\d+)\.out', f).group(1))
    if jid < 52240:
        continue
    task = int(re.search(r'mlp_sweep_(\d+)_', f).group(1))
    s_idx = task % 3; rem = task // 3
    rr_idx = rem % 3; rem //= 3
    wd_idx = rem % 3; h_idx = rem // 3
    if h_idx >= 3 or wd_idx >= 3 or rr_idx >= 3:
        continue
    key = (hidden_values[h_idx], wd_values[wd_idx], rr_values[rr_idx])
    with open(f) as fh:
        text = fh.read()
    epoch_lines = re.findall(r'test RMSE: ([\d.]+)', text)
    if epoch_lines and 'Done' in text:
        mlp_results.setdefault(key, []).append(float(epoch_lines[-1]))

ranked = sorted([{'h': h, 'wd': wd, 'rr': rr, 'mean': np.mean(v), 'std': np.std(v), 'n': len(v)}
                  for (h, wd, rr), v in mlp_results.items()], key=lambda x: x['mean'])
print(f"{'#':>2s} {'h':>3s} {'wd':>7s} {'rr':>6s} {'RMSE':>8s} {'±std':>7s} {'vs XRO':>8s}")
print("-" * 46)
for i, r in enumerate(ranked[:5]):
    vx = f"{(r['mean']-XRO)/XRO*100:+.1f}%"
    print(f"{i+1:2d} {r['h']:3d} {r['wd']:7.3f} {r['rr']:>6s} {r['mean']:8.4f} {r['std']:7.4f} {vx:>8s}")

# 4. SEASONAL GATE
csv_path = 'results_rebuttal_gate_ablation/seasonal_rmse_analysis.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("\n## 4. Seasonal Gate — Spring Barrier (MAM inits, leads 3-9)")
    mam = df[(df['season'] == 'MAM') & (df['lead'].between(3, 9))]
    summary = mam.groupby(['model', 'condition'])['rmse'].agg(['mean', 'std'])
    print(f"{'Model':15s} {'Condition':12s} {'MAM RMSE':>10s} {'±std':>7s} {'Δ gate':>10s}")
    print("-" * 58)
    for model in ['attentive', 'graph_pyg']:
        for cond in ['with_gate', 'no_gate']:
            if (model, cond) in summary.index:
                row = summary.loc[(model, cond)]
                delta = ''
                if cond == 'no_gate' and (model, 'with_gate') in summary.index:
                    wg = summary.loc[(model, 'with_gate'), 'mean']
                    delta = f"{row['mean'] - wg:+.4f}"
                print(f"{model:15s} {cond:12s} {row['mean']:10.4f} {row['std']:7.4f} {delta:>10s}")

    print("\n  Full seasonal breakdown:")
    all_s = df.groupby(['model', 'condition', 'season'])['rmse'].mean().unstack('season')
    print(all_s[['DJF', 'MAM', 'JJA', 'SON']].to_string(float_format=lambda x: f"{x:.3f}"))

# 5. DATA SCARCITY
models4 = ['linear', 'res', 'attentive', 'graph_pyg']
size_tags = ['10yr', '13yr', '16yr', '19yr', '23yr']
sc = {}
for f in sorted(glob.glob('slurm/logs/data_scarcity_*_*.out')):
    jid = int(re.search(r'_(\d+)\.out', f).group(1))
    if jid < 52065:
        continue
    task = int(re.search(r'data_scarcity_(\d+)_', f).group(1))
    m_idx = task // 15; rem = task % 15; sz_idx = rem // 3
    if m_idx >= 4 or sz_idx >= 5:
        continue
    model = models4[m_idx]; size = size_tags[sz_idx]
    with open(f) as fh:
        text = fh.read()
    epoch_lines = re.findall(r'test RMSE: ([\d.]+)', text)
    if epoch_lines and 'Done' in text:
        sc.setdefault((model, size), []).append(float(epoch_lines[-1]))

print(f"\n## 5. Data Scarcity Curve")
print(f"{'Model':15s}", end='')
for s in size_tags:
    print(f" {s:>14s}", end='')
print()
print("-" * (15 + 15 * len(size_tags)))
for model in models4:
    print(f"{model:15s}", end='')
    for size in size_tags:
        v = sc.get((model, size), [])
        if v:
            print(f" {np.mean(v):.3f}±{np.std(v):.3f}", end='')
        else:
            print(f" {'--':>14s}", end='')
    print()
print(f"{'XRO (ref)':15s}", end='')
for _ in size_tags:
    print(f" {'0.605':>14s}", end='')
print()

# 6. STOCHASTIC CRPS
print(f"\n## 6. Stochastic Ablation (CRPS)")
stoch_crps = {}
for model_dir in ['res', 'attentive', 'graphpyg']:
    dn = 'graph_pyg' if model_dir == 'graphpyg' else model_dir
    for noise in ['posthoc', 'stage2']:
        files = glob.glob(os.path.join(f'results_rebuttal_stochastic_ablation/{model_dir}',
                                        '**', f'*{noise}*seed*lead_metrics*'), recursive=True)
        crps = [pd.read_csv(f).query('lead > 0')['crps'].mean() for f in files]
        if crps:
            stoch_crps[(dn, noise)] = crps

print(f"{'Model':15s} {'Noise':10s} {'N':>3s} {'CRPS':>8s} {'±std':>7s} {'Δ':>8s}")
print("-" * 55)
for model in ['res', 'attentive', 'graph_pyg']:
    for noise in ['posthoc', 'stage2']:
        v = stoch_crps.get((model, noise), [])
        if v:
            delta = ''
            if noise == 'stage2':
                ph = stoch_crps.get((model, 'posthoc'), [])
                if ph:
                    delta = f"{np.mean(v)-np.mean(ph):+.4f}"
            print(f"{model:15s} {noise:10s} {len(v):3d} {np.mean(v):8.4f} {np.std(v):7.4f} {delta:>8s}")

print("\n" + "=" * 80)
print("END OF SUMMARY")
print("=" * 80)
