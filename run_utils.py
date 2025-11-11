"""
Utility functions for running NXRO model training and evaluation.
Contains run_*() functions for different model variants.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nxro.train import (
    train_nxro_linear,
    train_nxro_ro,
    train_nxro_rodiag,
    train_nxro_res,
    train_nxro_res_fullxro,
    train_nxro_neural,
    train_nxro_bilinear,
    train_nxro_attentive,
    train_nxro_graph,
    train_nxro_graph_pyg,
    train_nxro_neural_phys,
    train_nxro_resmix,
)
from utils.xro_utils import (
    calc_forecast_skill,
    nxro_reforecast,
    evaluate_stochastic_ensemble,
    plot_forecast_plume,
)
from nxro.nxro_utils import (
    simulate_nxro_longrun,
    plot_seasonal_sync,
    pick_sample_inits,
    evaluate_on_all_datasets_dual as evaluate_on_all_datasets,
    plot_skill_curves_multi_dataset,
    plot_skill_curves_dual,
)
from nxro.stochastic import (
    compute_residuals_series,
    fit_seasonal_ar1_from_residuals,
    SeasonalAR1Noise,
    nxro_reforecast_stochastic,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _evaluate_and_plot(fcst, obs_ds, train_period, test_period, eval_all_datasets, 
                       all_eval_datasets, base_dir, fig_suffix, sel_var, label):
    """Helper function to evaluate forecasts and generate plots."""
    if eval_all_datasets and all_eval_datasets:
        print("  Evaluating on all datasets...")
        all_results = evaluate_on_all_datasets(fcst, all_eval_datasets, train_period, test_period)
        acc_train = all_results['ORAS5']['acc_train']
        rmse_train = all_results['ORAS5']['rmse_train']
        acc_test = all_results['ORAS5']['acc_test']
        rmse_test = all_results['ORAS5']['rmse_test']
        plot_skill_curves_multi_dataset(all_results, sel_var, f'{base_dir}{fig_suffix}', label)
    else:
        acc_train = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                       by_month=False, verify_periods=train_period)
        rmse_train = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                        by_month=False, verify_periods=train_period)
        acc_test = calc_forecast_skill(fcst, obs_ds, metric='acc', is_mv3=True,
                                      by_month=False, verify_periods=test_period)
        rmse_test = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                       by_month=False, verify_periods=test_period)
    
    plot_skill_curves_dual(acc_train, rmse_train, acc_test, rmse_test,
                          sel_var=sel_var, out_prefix=f'{base_dir}{fig_suffix}', label=label)
    
    return acc_train, rmse_train, acc_test, rmse_test


def _run_stochastic_forecast(model, var_order, train_ds, obs_ds, args, base_dir, 
                            extra_tag, fig_suffix, train_period, test_period,
                            eval_all_datasets, all_eval_datasets, device, 
                            model_name, fcst_det=None):
    """Helper function to run stochastic ensemble forecasts."""
    print("  Generating stochastic ensemble forecasts...")
    
    # Determine noise source and suffix
    if args.use_sim_noise:
        print("  Fitting noise from simulation-observation differences...")
        from nxro.stochastic import fit_noise_from_simulations
        import glob
        sim_paths = glob.glob('data/XRO_indices_*_preproc.nc')
        a1_np, sigma_np = fit_noise_from_simulations(args.nc_path, sim_paths, var_order, train_period)
        noise_suffix = '_sim_noise'
    else:
        resid, months = compute_residuals_series(model, train_ds, var_order, device=device)
        a1_np, sigma_np = fit_seasonal_ar1_from_residuals(resid, months)
        noise_suffix = ''
    
    # Stage 2: Optimize noise parameters with likelihood if requested
    stage2_suffix = '_stage2' if args.train_noise_stage2 else ''
    combined_suffix = noise_suffix + stage2_suffix
    
    if args.train_noise_stage2:
        from nxro.stochastic import train_noise_stage2
        print("  Running Stage 2 noise optimization (likelihood-based)...")
        a1_np, sigma_np = train_noise_stage2(model, train_ds, var_order, 
                                             a1_np, sigma_np, 
                                             n_epochs=100, lr=1e-3, 
                                             device=device, verbose=True)
    
    a1 = torch.tensor(a1_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    noise = SeasonalAR1Noise(a1, sigma)
    
    NXRO_fcst_m = nxro_reforecast_stochastic(model, init_ds=obs_ds, n_month=21, var_order=var_order,
                                             noise_model=noise, n_members=args.members, device=device)
    
    # Save stochastic artifacts
    np.savez(f'{base_dir}/{model_name}_stochastic{combined_suffix}_noise{extra_tag}.npz', 
             a1=a1_np, sigma=sigma_np)
    torch.save({'state_dict': model.state_dict(), 'var_order': var_order, 'a1': a1.cpu(), 'sigma': sigma.cpu()},
              f'{base_dir}/{model_name}_stochastic{combined_suffix}{extra_tag}.pt')
    NXRO_fcst_m.to_netcdf(f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}_forecasts{extra_tag}.nc')
    
    # Ensemble evaluation
    evaluate_stochastic_ensemble(NXRO_fcst_m, obs_ds, var='Nino34', 
                                out_prefix=f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}_eval{extra_tag}')
    
    # Plume plots
    init_dates = pick_sample_inits(obs_ds, n=3)
    if len(init_dates) > 0 and fcst_det is not None:
        plot_forecast_plume(fcst_det, NXRO_fcst_m, obs_ds, init_dates, 
                           fname_prefix=f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}_plume', fig_suffix=fig_suffix)
    
    # Skills on ensemble mean
    NXRO_fcst_m_mean = NXRO_fcst_m.mean('member')
    _evaluate_and_plot(NXRO_fcst_m_mean, obs_ds, train_period, test_period, 
                      eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/{model_name.upper()}_stochastic{combined_suffix}', fig_suffix, 
                      'Nino34', f'{model_name.upper()} (stochastic mean)')


def run_linear(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, 
               load_xro_init, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Linear model."""
    base_dir = f'{base_results_dir}/linear'
    ensure_dir(base_dir)
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    L_basis_init = warmstart_params.get('L_basis_init') if warmstart_params else None
    
    model, var_order, best_rmse, history = train_nxro_linear(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc,
        L_basis_init=L_basis_init,
    )
    
    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Linear training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_linear_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    # Save weights
    lin_save = f'{base_dir}/nxro_linear{variant_suffix}_best{extra_tag}.pt'
    torch.save({'state_dict': model.state_dict(), 'var_order': var_order}, lin_save)
    print(f"✓ Saved to: {lin_save}")

    # Reforecast and evaluate
    NXRO_fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device=device)
    _evaluate_and_plot(NXRO_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_linear', fig_suffix, 'Nino34', 'NXRO-Linear')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(model, var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_linear', NXRO_fcst)
    
    # Seasonal synchronization
    sim_ds = simulate_nxro_longrun(model, X0_ds=train_ds, var_order=var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_linear_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Linear')
    
    print(f"✓ NXRO-Linear complete (out-of-sample)")


def run_ro(args, obs_ds, train_ds, test_ds, train_period, test_period, 
           base_results_dir, all_eval_datasets, device, 
           load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-RO model."""
    base_dir = f'{base_results_dir}/ro'
    ensure_dir(base_dir)
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=True, include_diag=False) if args.warm_start else None
    
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k != 'freeze_diag'}
    
    ro_model, ro_var_order, ro_best_rmse, ro_history = train_nxro_ro(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
    )
    
    # Training curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(ro_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(ro_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-RO training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_ro_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    # Save
    ro_save = f'{base_dir}/nxro_ro{variant_suffix}_best{extra_tag}.pt'
    torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order}, ro_save)
    print(f"✓ Saved to: {ro_save}")
    
    # Forecast and evaluate
    NXRO_ro_fcst = nxro_reforecast(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device=device)
    _evaluate_and_plot(NXRO_ro_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_ro', fig_suffix, 'Nino34', 'NXRO-RO')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(ro_model, ro_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_ro', NXRO_ro_fcst)
    
    ro_sim_ds = simulate_nxro_longrun(ro_model, X0_ds=train_ds, var_order=ro_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, ro_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_ro_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-RO')
    
    print(f"✓ NXRO-RO complete (out-of-sample)")


def run_rodiag(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, 
               load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-RO+Diag model."""
    base_dir = f'{base_results_dir}/rodiag'
    ensure_dir(base_dir)
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=True, include_diag=True) if args.warm_start else None
    
    rd_model, rd_var_order, rd_best_rmse, rd_history = train_nxro_rodiag(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags,
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rd_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rd_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-RO+Diag training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_rodiag_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rd_save = f'{base_dir}/nxro_rodiag{variant_suffix}_best{extra_tag}.pt'
    torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order}, rd_save)
    print(f"✓ Saved to: {rd_save}")
    
    NXRO_rd_fcst = nxro_reforecast(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device=device)
    _evaluate_and_plot(NXRO_rd_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_rodiag', fig_suffix, 'Nino34', 'NXRO-RO+Diag')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(rd_model, rd_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_rodiag', NXRO_rd_fcst)
    
    rd_sim_ds = simulate_nxro_longrun(rd_model, X0_ds=train_ds, var_order=rd_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rd_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_rodiag_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-RO+Diag')
    
    print(f"✓ NXRO-RO+Diag complete (out-of-sample)")


def run_res(args, obs_ds, train_ds, test_ds, train_period, test_period, 
            base_results_dir, all_eval_datasets, device, 
            load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Res model."""
    base_dir = f'{base_results_dir}/res'
    ensure_dir(base_dir)
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
    
    rs_model, rs_var_order, rs_best_rmse, rs_history = train_nxro_res(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rs_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rs_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Res training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_res_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rs_save = f'{base_dir}/nxro_res{variant_suffix}_best{extra_tag}.pt'
    torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order}, rs_save)
    print(f"✓ Saved to: {rs_save}")
    
    NXRO_rs_fcst = nxro_reforecast(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device=device)
    _evaluate_and_plot(NXRO_rs_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_res', fig_suffix, 'Nino34', 'NXRO-Res')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(rs_model, rs_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_res', NXRO_rs_fcst)
    
    rs_sim_ds = simulate_nxro_longrun(rs_model, X0_ds=train_ds, var_order=rs_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rs_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_res_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Res')
    
    print(f"✓ NXRO-Res complete (out-of-sample)")


def run_res_fullxro(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                    base_results_dir, all_eval_datasets, device, 
                    load_xro_init, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Res-FullXRO model."""
    assert args.warm_start is not None, "Variant res_fullxro requires --warm_start argument!"
    
    base_dir = f'{base_results_dir}/res_fullxro'
    ensure_dir(base_dir)
    
    xro_init = load_xro_init(args.warm_start, k_max=args.k_max, include_ro=True, include_diag=True)
    xro_init_dict = {k.replace('_init', ''): v for k, v in xro_init.items()}
    
    rs_fullxro_model, rs_fullxro_var_order, rs_fullxro_best_rmse, rs_fullxro_history = train_nxro_res_fullxro(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc,
        xro_init_dict=xro_init_dict
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rs_fullxro_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rs_fullxro_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Res-FullXRO training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_res_fullxro_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rs_fullxro_save = f'{base_dir}/nxro_res_fullxro_best{extra_tag}.pt'
    torch.save({'state_dict': rs_fullxro_model.state_dict(), 'var_order': rs_fullxro_var_order}, rs_fullxro_save)
    print(f"✓ Saved to: {rs_fullxro_save}")
    
    NXRO_rs_fullxro_fcst = nxro_reforecast(rs_fullxro_model, init_ds=obs_ds, n_month=21, 
                                           var_order=rs_fullxro_var_order, device=device)
    _evaluate_and_plot(NXRO_rs_fullxro_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_res_fullxro', fig_suffix, 'Nino34', 'NXRO-Res-FullXRO')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(rs_fullxro_model, rs_fullxro_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_res_fullxro', NXRO_rs_fullxro_fcst)
    
    rs_fullxro_sim_ds = simulate_nxro_longrun(rs_fullxro_model, X0_ds=train_ds, 
                                              var_order=rs_fullxro_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rs_fullxro_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_res_fullxro_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Res-FullXRO')
    
    print(f"✓ NXRO-Res-FullXRO complete (out-of-sample)")


def run_neural(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-NeuralODE model."""
    base_dir = f'{base_results_dir}/neural'
    ensure_dir(base_dir)
    
    nn_model, nn_var_order, nn_best_rmse, nn_history = train_nxro_neural(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only', 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(nn_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(nn_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-NeuralODE training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_neural_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    nn_save = f'{base_dir}/nxro_neural_best{extra_tag}.pt'
    torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order}, nn_save)
    print(f"✓ Saved to: {nn_save}")
    
    NXRO_nn_fcst = nxro_reforecast(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device=device)
    _evaluate_and_plot(NXRO_nn_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_neural', fig_suffix, 'Nino34', 'NXRO-NeuralODE')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(nn_model, nn_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_neural', NXRO_nn_fcst)
    
    nn_sim_ds = simulate_nxro_longrun(nn_model, X0_ds=train_ds, var_order=nn_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, nn_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_neural_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-NeuralODE')
    
    print(f"✓ NXRO-NeuralODE complete (out-of-sample)")


def run_neural_phys(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                    base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-PhysReg model."""
    base_dir = f'{base_results_dir}/neural_phys'
    ensure_dir(base_dir)
    
    np_model, np_var_order, np_best_rmse, np_history = train_nxro_neural_phys(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only',
        jac_reg=args.jac_reg, div_reg=args.div_reg, noise_std=args.noise_std, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(np_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(np_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-NeuralODE (PhysReg) training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_neural_phys_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    np_save = f'{base_dir}/nxro_neural_phys_best{extra_tag}.pt'
    torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order}, np_save)
    print(f"✓ Saved to: {np_save}")
    
    NXRO_np_fcst = nxro_reforecast(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order, device=device)
    _evaluate_and_plot(NXRO_np_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_neural_phys', fig_suffix, 'Nino34', 'NXRO-PhysReg')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(np_model, np_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_neural_phys', NXRO_np_fcst)
    
    np_sim_ds = simulate_nxro_longrun(np_model, X0_ds=train_ds, var_order=np_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, np_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_neural_phys_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-PhysReg')
    
    print(f"✓ NXRO-PhysReg complete (out-of-sample)")


def run_resmix(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, 
               load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-ResidualMix model."""
    base_dir = f'{base_results_dir}/resmix'
    ensure_dir(base_dir)
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=True, include_diag=True) if args.warm_start else None
    
    rx_model, rx_var_order, rx_best_rmse, rx_history = train_nxro_resmix(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, alpha_init=args.alpha_init, alpha_learnable=args.alpha_learnable,
        alpha_max=args.alpha_max, res_reg=args.res_reg, device=device,
        extra_train_nc_paths=args.extra_train_nc,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags,
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rx_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rx_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-ResidualMix training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_resmix_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rx_save = f'{base_dir}/nxro_resmix{variant_suffix}_best{extra_tag}.pt'
    torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order}, rx_save)
    print(f"✓ Saved to: {rx_save}")
    
    NXRO_rx_fcst = nxro_reforecast(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order, device=device)
    _evaluate_and_plot(NXRO_rx_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_resmix', fig_suffix, 'Nino34', 'NXRO-ResidualMix')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(rx_model, rx_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_resmix', NXRO_rx_fcst)
    
    rx_sim_ds = simulate_nxro_longrun(rx_model, X0_ds=train_ds, var_order=rx_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rx_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_resmix_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-ResidualMix')
    
    print(f"✓ NXRO-ResidualMix complete (out-of-sample)")


def run_bilinear(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                 base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Bilinear model."""
    base_dir = f'{base_results_dir}/bilinear'
    ensure_dir(base_dir)
    
    bl_model, bl_var_order, bl_best_rmse, bl_history = train_nxro_bilinear(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        n_channels=2, rank=2, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(bl_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(bl_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Bilinear training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_bilinear_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    bl_save = f'{base_dir}/nxro_bilinear_best{extra_tag}.pt'
    torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order}, bl_save)
    print(f"✓ Saved to: {bl_save}")
    
    NXRO_bl_fcst = nxro_reforecast(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device=device)
    _evaluate_and_plot(NXRO_bl_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_bilinear', fig_suffix, 'Nino34', 'NXRO-Bilinear')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(bl_model, bl_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_bilinear', NXRO_bl_fcst)
    
    bl_sim_ds = simulate_nxro_longrun(bl_model, X0_ds=train_ds, var_order=bl_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, bl_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_bilinear_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Bilinear')
    
    print(f"✓ NXRO-Bilinear complete (out-of-sample)")


def run_attentive(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                  base_results_dir, all_eval_datasets, device, 
                  load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Attentive model."""
    base_dir = f'{base_results_dir}/attentive'
    ensure_dir(base_dir)
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
    
    at_model, at_var_order, at_best_rmse, at_history = train_nxro_attentive(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        d=32, dropout=0.1, mask_mode='th_only', device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(at_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(at_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Attentive training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_attentive_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    at_save = f'{base_dir}/nxro_attentive{variant_suffix}_best{extra_tag}.pt'
    torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order}, at_save)
    print(f"✓ Saved to: {at_save}")
    
    NXRO_at_fcst = nxro_reforecast(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device=device)
    _evaluate_and_plot(NXRO_at_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_attentive', fig_suffix, 'Nino34', 'NXRO-Attentive')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(at_model, at_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_attentive', NXRO_at_fcst)
    
    at_sim_ds = simulate_nxro_longrun(at_model, X0_ds=train_ds, var_order=at_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, at_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_attentive_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Attentive')
    
    print(f"✓ NXRO-Attentive complete (out-of-sample)")


def run_graph(args, obs_ds, train_ds, test_ds, train_period, test_period, 
              base_results_dir, all_eval_datasets, device, 
              load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Graph model."""
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
    
    gr_model, gr_var_order, gr_best_rmse, gr_history = train_nxro_graph(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        use_fixed_graph=(not args.graph_learned),
        learned_l1_lambda=args.graph_l1,
        stat_knn_method=args.graph_stat_method,
        stat_knn_top_k=args.graph_stat_topk,
        stat_knn_source=args.graph_stat_source,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc
    )
    
    graph_kind = f"stat_{args.graph_stat_method}_k{args.graph_stat_topk}" if args.graph_stat_method else "xro"
    graph_mode = "learned" if args.graph_learned else "fixed"
    l1_tag = f"_l1{args.graph_l1}" if args.graph_learned and args.graph_l1 > 0 else ""
    graph_tag = f"_{graph_mode}_{graph_kind}{l1_tag}"
    base_dir = f"{base_results_dir}/graph/{graph_mode}_{graph_kind}{l1_tag}"
    ensure_dir(base_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(gr_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(gr_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f'NXRO-Graph ({graph_mode}, {graph_kind}) training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}_training_curves.png', dpi=300)
    plt.close()
    
    gr_save = f'{base_dir}/nxro_graph{graph_tag}_best.pt'
    torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order}, gr_save)
    print(f"✓ Saved to: {gr_save}")
    
    NXRO_gr_fcst = nxro_reforecast(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device=device)
    _evaluate_and_plot(NXRO_gr_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_graph{graph_tag}', fig_suffix, 'Nino34', 'NXRO-Graph')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(gr_model, gr_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                f'nxro_graph{graph_tag}', NXRO_gr_fcst)
    
    gr_sim_ds = simulate_nxro_longrun(gr_model, X0_ds=train_ds, var_order=gr_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, gr_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_graph{graph_tag}_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Graph')
    
    print(f"✓ NXRO-Graph complete (out-of-sample)")


def run_graph_pyg(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                  base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-GraphPyG model."""
    gp_model, gp_var_order, gp_best_rmse, gp_history = train_nxro_graph_pyg(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        top_k=args.top_k, hidden=16, dropout=0.1, use_gat=args.gat, device=device, 
        rollout_k=args.rollout_k,
        extra_train_nc_paths=args.extra_train_nc
    )
    
    tag2 = 'gat' if args.gat else 'gcn'
    ktag = f"k{args.top_k}"
    base_dir = f'{base_results_dir}/graphpyg/{tag2}_{ktag}'
    ensure_dir(base_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(gp_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(gp_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f'NXRO-GraphPyG ({tag2}, {ktag}) training (Out-of-Sample)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_training_curves.png', dpi=300)
    plt.close()
    
    gp_save = f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}_best{extra_tag}.pt'
    torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order}, gp_save)
    print(f"✓ Saved to: {gp_save}")
    
    NXRO_gp_fcst = nxro_reforecast(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order, device=device)
    _evaluate_and_plot(NXRO_gp_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}', fig_suffix, 'Nino34', 
                      f'NXRO-GraphPyG ({tag2.upper()})')
    
    # Stochastic ensemble forecasts (optional)
    if args.stochastic:
        _run_stochastic_forecast(gp_model, gp_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                f'nxro_graphpyg_{tag2}_{ktag}', NXRO_gp_fcst)
    
    gp_sim_ds = simulate_nxro_longrun(gp_model, X0_ds=train_ds, var_order=gp_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, gp_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_seasonal_synchronization.png',
                      model_label=f'NXRO-GraphPyG ({tag2.upper()})')
    
    print(f"✓ NXRO-GraphPyG complete (out-of-sample)")

