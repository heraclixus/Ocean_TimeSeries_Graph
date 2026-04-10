"""Run LSTM and GRU baselines for ENSO forecasting.

Simple sequence-to-one models: given the current state X_t, predict X_{t+1}.
Then Euler-step for multi-lead forecasting (same protocol as NXRO).

Usage:
    python scripts/run_lstm_gru_baselines.py --device cuda --seeds 42 123 256
"""
import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import xarray as xr

from nxro.data import get_dataloaders
from utils.xro_utils import nxro_reforecast, calc_forecast_skill


class LSTMForecaster(nn.Module):
    def __init__(self, n_vars, hidden=32, n_layers=1, dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.lstm = nn.LSTM(n_vars, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n_vars)

    def forward(self, x, t_years=None):
        # x: [B, n_vars] — treat as single-step sequence [B, 1, n_vars]
        out, _ = self.lstm(x.unsqueeze(1))
        dxdt = self.fc(out.squeeze(1))
        return dxdt


class GRUForecaster(nn.Module):
    def __init__(self, n_vars, hidden=32, n_layers=1, dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.gru = nn.GRU(n_vars, hidden, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n_vars)

    def forward(self, x, t_years=None):
        out, _ = self.gru(x.unsqueeze(1))
        dxdt = self.fc(out.squeeze(1))
        return dxdt


def train_model(model, dl_train, dl_val, dl_test, n_epochs, lr, weight_decay, device, tag):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    dt = 1.0 / 12.0

    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0

    dl_select = dl_val if dl_val is not None else dl_test

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss, train_n = 0.0, 0
        for batch in dl_train:
            x_t, t_y, x_next = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            dxdt = model(x_t, t_y)
            x_hat = x_t + dxdt * dt
            loss = loss_fn(x_hat, x_next)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * x_t.size(0)
            train_n += x_t.size(0)

        # Eval on selection set
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in dl_select:
                x_t, t_y, x_next = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                dxdt = model(x_t, t_y)
                x_hat = x_t + dxdt * dt
                loss = loss_fn(x_hat, x_next)
                val_loss += loss.item() * x_t.size(0)
                val_n += x_t.size(0)

        train_rmse = (train_loss / max(train_n, 1)) ** 0.5
        val_rmse = (val_loss / max(val_n, 1)) ** 0.5

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"[{tag}] Epoch {epoch:04d} | train: {train_rmse:.4f} | val: {val_rmse:.4f} | best: {best_rmse:.4f} (ep {best_epoch})")

        if patience_counter >= 200:
            print(f"[{tag}] Early stop at epoch {epoch}, best: {best_rmse:.4f} (ep {best_epoch})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_rmse, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 256])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--out_dir', default='results_rebuttal_lstm_gru')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    obs_ds = xr.open_dataset('data/XRO_indices_oras5.nc')
    test_period = slice('2002-01', '2022-12')

    configs = [
        ('LSTM_h32', LSTMForecaster, {'hidden': 32, 'n_layers': 1}),
        ('LSTM_h64', LSTMForecaster, {'hidden': 64, 'n_layers': 1}),
        ('GRU_h32', GRUForecaster, {'hidden': 32, 'n_layers': 1}),
        ('GRU_h64', GRUForecaster, {'hidden': 64, 'n_layers': 1}),
    ]

    all_results = []

    for name, model_cls, model_kwargs in configs:
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load data with val split
            result = get_dataloaders(
                nc_path='data/XRO_indices_oras5.nc',
                train_slice=('1979-01', '1995-12'),
                val_slice=('1996-01', '2001-12'),
                test_slice=('2002-01', '2022-12'),
                batch_size=128,
            )
            if len(result) == 4:
                dl_train, dl_val, dl_test, var_order = result
            else:
                dl_train, dl_test, var_order = result
                dl_val = None

            n_vars = len(var_order)
            model = model_cls(n_vars=n_vars, **model_kwargs).to(device)
            n_params = sum(p.numel() for p in model.parameters())

            tag = f"{name}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"{tag} ({n_params} params)")
            print(f"{'='*60}")

            model, best_rmse, best_epoch = train_model(
                model, dl_train, dl_val, dl_test,
                n_epochs=args.epochs, lr=1e-3, weight_decay=1e-3,
                device=device, tag=tag,
            )

            # Reforecast and compute skill
            model.eval()
            fcst = nxro_reforecast(model, init_ds=obs_ds, n_month=21,
                                   var_order=var_order, device=device)
            rmse_skill = calc_forecast_skill(fcst, obs_ds, metric='rmse', is_mv3=True,
                                             by_month=False, verify_periods=test_period)
            nino34_rmse = rmse_skill['Nino34'].values
            avg_rmse = nino34_rmse[1:].mean()  # skip lead 0

            print(f"[{tag}] Nino3.4 avg RMSE (leads 1-21): {avg_rmse:.4f}")

            all_results.append({
                'model': name, 'seed': seed, 'params': n_params,
                'val_rmse_1step': float(best_rmse),
                'nino34_avg_rmse': float(avg_rmse),
                'best_epoch': best_epoch,
            })

    # Summary
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.out_dir, 'lstm_gru_results.csv'), index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = df.groupby('model').agg(
        n=('seed', 'count'),
        val_rmse_mean=('val_rmse_1step', 'mean'),
        val_rmse_std=('val_rmse_1step', 'std'),
        nino34_mean=('nino34_avg_rmse', 'mean'),
        nino34_std=('nino34_avg_rmse', 'std'),
        params=('params', 'first'),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nReference: XRO = 0.605, NXRO-Attentive = 0.555, NXRO-GNN = 0.557")


if __name__ == '__main__':
    main()
