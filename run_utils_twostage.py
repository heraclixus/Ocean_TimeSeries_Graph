"""
Utility functions for Two-Stage NXRO model training (Synthetic Pre-training -> ORAS5 Fine-tuning).
Derived from run_utils.py.
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
    train_nxro_transformer,
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
    fit_seasonal_arp_from_residuals,
    SeasonalAR1Noise,
    SeasonalARPNoise,
    nxro_reforecast_stochastic,
    nxro_reforecast_stochastic_arp,
)
from run_utils import ensure_dir, _evaluate_and_plot, _run_stochastic_forecast


def run_linear_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, 
               load_xro_init, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Linear model (Two-Stage)."""
    base_dir = f'{base_results_dir}/linear'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")

    # --- Stage 1: Synthetic Pre-training ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    L_basis_init = warmstart_params.get('L_basis_init') if warmstart_params else None

    # Note: Using args.train_start/end for synthetic data slicing. 
    # Assumes synthetic data covers this period or is aligned.
    model_s1, _, _, _ = train_nxro_linear(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra,
        L_basis_init=L_basis_init,
    )
    
    s1_path = f'{base_dir}/nxro_linear_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()

    # --- Stage 2: Fine-tuning on ORAS5 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    model, var_order, best_rmse, history = train_nxro_linear(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None, # Fine-tune ONLY on ORAS5
        L_basis_init=L_basis_init,
        pretrained_state_dict=stage1_weights
    )
    
    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Linear training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_linear_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    # Save weights
    lin_save = f'{base_dir}/nxro_linear{variant_suffix}_real_finetuned.pt'
    torch.save({'state_dict': model.state_dict(), 'var_order': var_order}, lin_save)
    print(f"✓ Saved to: {lin_save}")

    # Reforecast and evaluate
    NXRO_fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21, var_order=var_order, device=device)
    _evaluate_and_plot(NXRO_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_linear', fig_suffix, 'Nino34', 'NXRO-Linear (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(model, var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_linear', NXRO_fcst)
    
    sim_ds = simulate_nxro_longrun(model, X0_ds=train_ds, var_order=var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_linear_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Linear')
    print(f"✓ NXRO-Linear complete (Two-Stage)")
    

def run_ro_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
           base_results_dir, all_eval_datasets, device, 
           load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-RO model (Two-Stage)."""
    base_dir = f'{base_results_dir}/ro'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=True, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k != 'freeze_diag'}
    
    model_s1, _, _, _ = train_nxro_ro(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
    )
    s1_path = f'{base_dir}/nxro_ro_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    ro_model, ro_var_order, ro_best_rmse, ro_history = train_nxro_ro(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(ro_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(ro_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-RO training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_ro_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    ro_save = f'{base_dir}/nxro_ro{variant_suffix}_real_finetuned.pt'
    torch.save({'state_dict': ro_model.state_dict(), 'var_order': ro_var_order}, ro_save)
    print(f"✓ Saved to: {ro_save}")
    
    NXRO_ro_fcst = nxro_reforecast(ro_model, init_ds=obs_ds, n_month=21, var_order=ro_var_order, device=device)
    _evaluate_and_plot(NXRO_ro_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_ro', fig_suffix, 'Nino34', 'NXRO-RO (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(ro_model, ro_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_ro', NXRO_ro_fcst)
    
    ro_sim_ds = simulate_nxro_longrun(ro_model, X0_ds=train_ds, var_order=ro_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, ro_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_ro_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-RO')
    print(f"✓ NXRO-RO complete (Two-Stage)")
    

def run_rodiag_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, 
               load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-RO+Diag model (Two-Stage)."""
    base_dir = f'{base_results_dir}/rodiag'
    ensure_dir(base_dir)
    
    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")

    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=True, include_diag=True) if args.warm_start else None
    
    model_s1, _, _, _ = train_nxro_rodiag(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags,
    )
    s1_path = f'{base_dir}/nxro_rodiag_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    rd_model, rd_var_order, rd_best_rmse, rd_history = train_nxro_rodiag(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rd_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rd_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-RO+Diag training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_rodiag_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rd_save = f'{base_dir}/nxro_rodiag{variant_suffix}_real_finetuned.pt'
    torch.save({'state_dict': rd_model.state_dict(), 'var_order': rd_var_order}, rd_save)
    print(f"✓ Saved to: {rd_save}")
    
    NXRO_rd_fcst = nxro_reforecast(rd_model, init_ds=obs_ds, n_month=21, var_order=rd_var_order, device=device)
    _evaluate_and_plot(NXRO_rd_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_rodiag', fig_suffix, 'Nino34', 'NXRO-RO+Diag (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(rd_model, rd_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_rodiag', NXRO_rd_fcst)
    
    rd_sim_ds = simulate_nxro_longrun(rd_model, X0_ds=train_ds, var_order=rd_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rd_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_rodiag_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-RO+Diag')
    print(f"✓ NXRO-RO+Diag complete (Two-Stage)")
    

def run_res_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
            base_results_dir, all_eval_datasets, device, 
            load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Res model (Two-Stage)."""
    base_dir = f'{base_results_dir}/res'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
    
    model_s1, _, _, _ = train_nxro_res(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
    )
    s1_path = f'{base_dir}/nxro_res_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    rs_model, rs_var_order, rs_best_rmse, rs_history = train_nxro_res(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rs_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rs_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Res training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_res_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rs_save = f'{base_dir}/nxro_res{variant_suffix}_real_finetuned.pt'
    torch.save({'state_dict': rs_model.state_dict(), 'var_order': rs_var_order}, rs_save)
    print(f"✓ Saved to: {rs_save}")
    
    NXRO_rs_fcst = nxro_reforecast(rs_model, init_ds=obs_ds, n_month=21, var_order=rs_var_order, device=device)
    _evaluate_and_plot(NXRO_rs_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_res', fig_suffix, 'Nino34', 'NXRO-Res (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(rs_model, rs_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_res', NXRO_rs_fcst)
    
    rs_sim_ds = simulate_nxro_longrun(rs_model, X0_ds=train_ds, var_order=rs_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rs_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_res_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Res')
    print(f"✓ NXRO-Res complete (Two-Stage)")
    

def run_res_fullxro_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                    base_results_dir, all_eval_datasets, device, 
                    load_xro_init, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Res-FullXRO model (Two-Stage)."""
    assert args.warm_start is not None, "Variant res_fullxro requires --warm_start argument!"
    
    base_dir = f'{base_results_dir}/res_fullxro'
    ensure_dir(base_dir)
    
    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    xro_init = load_xro_init(args.warm_start, k_max=args.k_max, include_ro=True, include_diag=True)
    xro_init_dict = {k.replace('_init', ''): v for k, v in xro_init.items()}
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_res_fullxro(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra,
        xro_init_dict=xro_init_dict
    )
    s1_path = f'{base_dir}/nxro_res_fullxro_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    rs_fullxro_model, rs_fullxro_var_order, rs_fullxro_best_rmse, rs_fullxro_history = train_nxro_res_fullxro(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        res_reg=args.res_reg, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        xro_init_dict=xro_init_dict,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rs_fullxro_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rs_fullxro_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Res-FullXRO training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_res_fullxro_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rs_fullxro_save = f'{base_dir}/nxro_res_fullxro_real_finetuned.pt'
    torch.save({'state_dict': rs_fullxro_model.state_dict(), 'var_order': rs_fullxro_var_order}, rs_fullxro_save)
    print(f"✓ Saved to: {rs_fullxro_save}")
    
    NXRO_rs_fullxro_fcst = nxro_reforecast(rs_fullxro_model, init_ds=obs_ds, n_month=21, 
                                           var_order=rs_fullxro_var_order, device=device)
    _evaluate_and_plot(NXRO_rs_fullxro_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_res_fullxro', fig_suffix, 'Nino34', 'NXRO-Res-FullXRO (Two-Stage)')
    
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
    print(f"✓ NXRO-Res-FullXRO complete (Two-Stage)")
    

def run_neural_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-NeuralODE model (Two-Stage)."""
    base_dir = f'{base_results_dir}/neural'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_neural(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only', 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra
    )
    s1_path = f'{base_dir}/nxro_neural_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    nn_model, nn_var_order, nn_best_rmse, nn_history = train_nxro_neural(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only', 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(nn_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(nn_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-NeuralODE training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_neural_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    nn_save = f'{base_dir}/nxro_neural_real_finetuned.pt'
    torch.save({'state_dict': nn_model.state_dict(), 'var_order': nn_var_order}, nn_save)
    print(f"✓ Saved to: {nn_save}")
    
    NXRO_nn_fcst = nxro_reforecast(nn_model, init_ds=obs_ds, n_month=21, var_order=nn_var_order, device=device)
    _evaluate_and_plot(NXRO_nn_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_neural', fig_suffix, 'Nino34', 'NXRO-NeuralODE (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(nn_model, nn_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_neural', NXRO_nn_fcst)
    
    nn_sim_ds = simulate_nxro_longrun(nn_model, X0_ds=train_ds, var_order=nn_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, nn_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_neural_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-NeuralODE')
    print(f"✓ NXRO-NeuralODE complete (Two-Stage)")
    

def run_neural_phys_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                    base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-PhysReg model (Two-Stage)."""
    base_dir = f'{base_results_dir}/neural_phys'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_neural_phys(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only',
        jac_reg=args.jac_reg, div_reg=args.div_reg, noise_std=args.noise_std, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra
    )
    s1_path = f'{base_dir}/nxro_neural_phys_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    np_model, np_var_order, np_best_rmse, np_history = train_nxro_neural_phys(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, depth=2, dropout=0.1, allow_cross=False, mask_mode='th_only',
        jac_reg=args.jac_reg, div_reg=args.div_reg, noise_std=args.noise_std, 
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(np_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(np_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-NeuralODE (PhysReg) training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_neural_phys_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    np_save = f'{base_dir}/nxro_neural_phys_real_finetuned.pt'
    torch.save({'state_dict': np_model.state_dict(), 'var_order': np_var_order}, np_save)
    print(f"✓ Saved to: {np_save}")
    
    NXRO_np_fcst = nxro_reforecast(np_model, init_ds=obs_ds, n_month=21, var_order=np_var_order, device=device)
    _evaluate_and_plot(NXRO_np_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_neural_phys', fig_suffix, 'Nino34', 'NXRO-PhysReg (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(np_model, np_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_neural_phys', NXRO_np_fcst)
    
    np_sim_ds = simulate_nxro_longrun(np_model, X0_ds=train_ds, var_order=np_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, np_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_neural_phys_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-PhysReg')
    print(f"✓ NXRO-PhysReg complete (Two-Stage)")
    

def run_resmix_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
               base_results_dir, all_eval_datasets, device, 
               load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-ResidualMix model (Two-Stage)."""
    base_dir = f'{base_results_dir}/resmix'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=True, include_diag=True) if args.warm_start else None
    
    model_s1, _, _, _ = train_nxro_resmix(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, alpha_init=args.alpha_init, alpha_learnable=args.alpha_learnable,
        alpha_max=args.alpha_max, res_reg=args.res_reg, device=device,
        extra_train_nc_paths=stage1_extra,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags,
    )
    s1_path = f'{base_dir}/nxro_resmix_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    rx_model, rx_var_order, rx_best_rmse, rx_history = train_nxro_resmix(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        hidden=64, alpha_init=args.alpha_init, alpha_learnable=args.alpha_learnable,
        alpha_max=args.alpha_max, res_reg=args.res_reg, device=device,
        extra_train_nc_paths=None,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(rx_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(rx_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-ResidualMix training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_resmix_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    rx_save = f'{base_dir}/nxro_resmix{variant_suffix}_real_finetuned.pt'
    torch.save({'state_dict': rx_model.state_dict(), 'var_order': rx_var_order}, rx_save)
    print(f"✓ Saved to: {rx_save}")
    
    NXRO_rx_fcst = nxro_reforecast(rx_model, init_ds=obs_ds, n_month=21, var_order=rx_var_order, device=device)
    _evaluate_and_plot(NXRO_rx_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_resmix', fig_suffix, 'Nino34', 'NXRO-ResidualMix (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(rx_model, rx_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_resmix', NXRO_rx_fcst)
    
    rx_sim_ds = simulate_nxro_longrun(rx_model, X0_ds=train_ds, var_order=rx_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, rx_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_resmix_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-ResidualMix')
    print(f"✓ NXRO-ResidualMix complete (Two-Stage)")
    

def run_bilinear_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                 base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Bilinear model (Two-Stage)."""
    base_dir = f'{base_results_dir}/bilinear'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_bilinear(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        n_channels=2, rank=2, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra
    )
    s1_path = f'{base_dir}/nxro_bilinear_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    bl_model, bl_var_order, bl_best_rmse, bl_history = train_nxro_bilinear(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        n_channels=2, rank=2, device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(bl_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(bl_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Bilinear training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_bilinear_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    bl_save = f'{base_dir}/nxro_bilinear_real_finetuned.pt'
    torch.save({'state_dict': bl_model.state_dict(), 'var_order': bl_var_order}, bl_save)
    print(f"✓ Saved to: {bl_save}")
    
    NXRO_bl_fcst = nxro_reforecast(bl_model, init_ds=obs_ds, n_month=21, var_order=bl_var_order, device=device)
    _evaluate_and_plot(NXRO_bl_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_bilinear', fig_suffix, 'Nino34', 'NXRO-Bilinear (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(bl_model, bl_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_bilinear', NXRO_bl_fcst)
    
    bl_sim_ds = simulate_nxro_longrun(bl_model, X0_ds=train_ds, var_order=bl_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, bl_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_bilinear_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Bilinear')
    print(f"✓ NXRO-Bilinear complete (Two-Stage)")
    

def run_attentive_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                  base_results_dir, all_eval_datasets, device, 
                  load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Attentive model (Two-Stage)."""
    base_dir = f'{base_results_dir}/attentive'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
    
    model_s1, _, _, _ = train_nxro_attentive(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        d=32, dropout=0.1, mask_mode='th_only', device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
    )
    s1_path = f'{base_dir}/nxro_attentive_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    at_model, at_var_order, at_best_rmse, at_history = train_nxro_attentive(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        d=32, dropout=0.1, mask_mode='th_only', device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        warmstart_init_dict=warmstart_params,
        freeze_flags=freeze_flags_filtered,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(at_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(at_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Attentive training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_attentive_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    at_save = f'{base_dir}/nxro_attentive{variant_suffix}_real_finetuned.pt'
    torch.save({'state_dict': at_model.state_dict(), 'var_order': at_var_order}, at_save)
    print(f"✓ Saved to: {at_save}")
    
    NXRO_at_fcst = nxro_reforecast(at_model, init_ds=obs_ds, n_month=21, var_order=at_var_order, device=device)
    _evaluate_and_plot(NXRO_at_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_attentive', fig_suffix, 'Nino34', 'NXRO-Attentive (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(at_model, at_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_attentive', NXRO_at_fcst)
    
    at_sim_ds = simulate_nxro_longrun(at_model, X0_ds=train_ds, var_order=at_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, at_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_attentive_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Attentive')
    print(f"✓ NXRO-Attentive complete (Two-Stage)")
    

def run_graph_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
              base_results_dir, all_eval_datasets, device, 
              load_xro_init, freeze_flags, variant_suffix, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Graph model (Two-Stage)."""
    warmstart_params = load_xro_init(args.warm_start, k_max=args.k_max, 
                                     include_ro=False, include_diag=False) if args.warm_start else None
    freeze_flags_filtered = {k: v for k, v in freeze_flags.items() if k == 'freeze_linear'}
    
    graph_kind = f"stat_{args.graph_stat_method}_k{args.graph_stat_topk}" if args.graph_stat_method else "xro"
    graph_mode = "learned" if args.graph_learned else "fixed"
    l1_tag = f"_l1{args.graph_l1}" if args.graph_learned and args.graph_l1 > 0 else ""
    graph_tag = f"_{graph_mode}_{graph_kind}{l1_tag}"
    base_dir = f"{base_results_dir}/graph/{graph_mode}_{graph_kind}{l1_tag}"
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_graph(
        nc_path=stage1_nc_path,
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
        extra_train_nc_paths=stage1_extra
    )
    s1_path = f'{base_dir}/nxro_graph{graph_tag}_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
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
        extra_train_nc_paths=None,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(gr_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(gr_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f'NXRO-Graph ({graph_mode}, {graph_kind}) training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_graph{graph_tag}{fig_suffix}_training_curves.png', dpi=300)
    plt.close()
    
    gr_save = f'{base_dir}/nxro_graph{graph_tag}_real_finetuned.pt'
    torch.save({'state_dict': gr_model.state_dict(), 'var_order': gr_var_order}, gr_save)
    print(f"✓ Saved to: {gr_save}")
    
    NXRO_gr_fcst = nxro_reforecast(gr_model, init_ds=obs_ds, n_month=21, var_order=gr_var_order, device=device)
    _evaluate_and_plot(NXRO_gr_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_graph{graph_tag}', fig_suffix, 'Nino34', 'NXRO-Graph (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(gr_model, gr_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                f'nxro_graph{graph_tag}', NXRO_gr_fcst)
    
    gr_sim_ds = simulate_nxro_longrun(gr_model, X0_ds=train_ds, var_order=gr_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, gr_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_graph{graph_tag}_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Graph')
    print(f"✓ NXRO-Graph complete (Two-Stage)")
    

def run_graph_pyg_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                  base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-GraphPyG model (Two-Stage)."""
    tag2 = 'gat' if args.gat else 'gcn'
    ktag = f"k{args.top_k}"
    base_dir = f'{base_results_dir}/graphpyg/{tag2}_{ktag}'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_graph_pyg(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        top_k=args.top_k, hidden=16, dropout=0.1, use_gat=args.gat, device=device, 
        rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra
    )
    s1_path = f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    gp_model, gp_var_order, gp_best_rmse, gp_history = train_nxro_graph_pyg(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        top_k=args.top_k, hidden=16, dropout=0.1, use_gat=args.gat, device=device, 
        rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(gp_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(gp_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f'NXRO-GraphPyG ({tag2}, {ktag}) training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_training_curves.png', dpi=300)
    plt.close()
    
    gp_save = f'{base_dir}/nxro_graphpyg_{tag2}_{ktag}_real_finetuned.pt'
    torch.save({'state_dict': gp_model.state_dict(), 'var_order': gp_var_order}, gp_save)
    print(f"✓ Saved to: {gp_save}")
    
    NXRO_gp_fcst = nxro_reforecast(gp_model, init_ds=obs_ds, n_month=21, var_order=gp_var_order, device=device)
    _evaluate_and_plot(NXRO_gp_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}', fig_suffix, 'Nino34', 
                      f'NXRO-GraphPyG ({tag2.upper()}) (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(gp_model, gp_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                f'nxro_graphpyg_{tag2}_{ktag}', NXRO_gp_fcst)
    
    gp_sim_ds = simulate_nxro_longrun(gp_model, X0_ds=train_ds, var_order=gp_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, gp_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_graphpyg_{tag2}_{ktag}{fig_suffix}_seasonal_synchronization.png',
                      model_label=f'NXRO-GraphPyG ({tag2.upper()})')
    print(f"✓ NXRO-GraphPyG complete (Two-Stage)")


def run_transformer_twostage(args, obs_ds, train_ds, test_ds, train_period, test_period, 
                              base_results_dir, all_eval_datasets, device, extra_tag, fig_suffix):
    """Train and evaluate NXRO-Transformer model (Two-Stage)."""
    base_dir = f'{base_results_dir}/transformer'
    ensure_dir(base_dir)

    if not args.extra_train_nc:
        raise ValueError("Two-stage training requires synthetic datasets via --extra_train_nc!")
    
    # --- Stage 1 ---
    print("\n" + "="*40)
    print("STAGE 1: Synthetic Pre-training")
    print("="*40)
    stage1_nc_path = args.extra_train_nc[0]
    stage1_extra = args.extra_train_nc[1:]
    
    model_s1, _, _, _ = train_nxro_transformer(
        nc_path=stage1_nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=stage1_extra
    )
    s1_path = f'{base_dir}/nxro_transformer_synthetic_pretrained.pt'
    torch.save(model_s1.state_dict(), s1_path)
    print(f"✓ Saved Stage 1 model to: {s1_path}")
    stage1_weights = model_s1.state_dict()
    
    # --- Stage 2 ---
    print("\n" + "="*40)
    print("STAGE 2: Fine-tuning on ORAS5")
    print("="*40)
    
    tf_model, tf_var_order, tf_best_rmse, tf_history = train_nxro_transformer(
        nc_path=args.nc_path,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, k_max=args.k_max,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
        device=device, rollout_k=args.rollout_k,
        extra_train_nc_paths=None,
        pretrained_state_dict=stage1_weights
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(tf_history['train_rmse'], label='train RMSE', c='tab:blue')
    ax.plot(tf_history['test_rmse'], label='test RMSE (out-of-sample)', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('NXRO-Transformer training (Out-of-Sample, Two-Stage)')
    ax.legend()
    plt.savefig(f'{base_dir}/NXRO_transformer_training_curves{fig_suffix}.png', dpi=300)
    plt.close()
    
    tf_save = f'{base_dir}/nxro_transformer_real_finetuned.pt'
    torch.save({'state_dict': tf_model.state_dict(), 'var_order': tf_var_order}, tf_save)
    print(f"✓ Saved to: {tf_save}")
    
    NXRO_tf_fcst = nxro_reforecast(tf_model, init_ds=obs_ds, n_month=21, var_order=tf_var_order, device=device)
    _evaluate_and_plot(NXRO_tf_fcst, obs_ds, train_period, test_period, 
                      args.eval_all_datasets, all_eval_datasets,
                      f'{base_dir}/NXRO_transformer', fig_suffix, 'Nino34', 'NXRO-Transformer (Two-Stage)')
    
    if args.stochastic:
        _run_stochastic_forecast(tf_model, tf_var_order, train_ds, obs_ds, args, base_dir, 
                                extra_tag, fig_suffix, train_period, test_period,
                                args.eval_all_datasets, all_eval_datasets, device, 
                                'nxro_transformer', NXRO_tf_fcst)
    
    tf_sim_ds = simulate_nxro_longrun(tf_model, X0_ds=train_ds, var_order=tf_var_order, nyear=100, device=device)
    plot_seasonal_sync(train_ds, tf_sim_ds, sel_var='Nino34', 
                      out_path=f'{base_dir}/NXRO_transformer_seasonal_synchronization{fig_suffix}.png',
                      model_label='NXRO-Transformer')
    print(f"✓ NXRO-Transformer complete (Two-Stage)")
