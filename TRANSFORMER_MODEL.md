# NXRO-Transformer Model

## Overview

The NXRO-Transformer is a pure Transformer-based model for ENSO forecasting that follows the same Neural ODE framework as other NXRO models. It uses multi-head self-attention to capture complex interactions between climate variables.

## Architecture

### Model Equation

$$
\frac{d\mathbf{X}}{dt} = \text{Transformer}_\theta(\mathbf{X}, \boldsymbol{\phi}(t))
$$

where:
- $\mathbf{X} \in \mathbb{R}^{n}$ is the state vector (climate variables)
- $\boldsymbol{\phi}(t) = [1, \cos(\omega t), \sin(\omega t), \cos(2\omega t), \sin(2\omega t)]^T$ are seasonal Fourier features
- $\text{Transformer}_\theta$ is a Transformer encoder network

### Components

1. **Input Projection**
   - Concatenates state variables with seasonal time features
   - Projects to `d_model` dimensions
   - Input: `[state (n_vars) + time_features (1 + 2*k_max)]` → Output: `d_model`

2. **Variable Positional Encoding**
   - Learnable positional embeddings for each variable
   - Treats each climate variable as a token in the sequence
   - Allows the model to distinguish between different variables

3. **Transformer Encoder**
   - Multi-head self-attention mechanism
   - Captures interactions between all climate variables
   - Default: 2 layers, 4 attention heads
   - Feedforward dimension: 256
   - Activation: GELU

4. **Output Projection**
   - Maps from `d_model` back to derivatives for each variable
   - Two-layer MLP with GELU activation
   - Output: `dX/dt` for each variable

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Transformer embedding dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 2 | Number of transformer encoder layers |
| `dim_feedforward` | 256 | Dimension of feedforward network |
| `dropout` | 0.1 | Dropout rate |
| `k_max` | 2 | Maximum harmonic order for seasonal features |

## Advantages

1. **State-Dependent Interactions**: Self-attention allows the model to learn which variables influence each other dynamically based on the current state.

2. **No Architectural Bias**: Unlike physics-based models, the Transformer learns all interactions from data without imposing specific polynomial or linear structures.

3. **Scalability**: Transformer architecture scales well with the number of variables and can capture long-range dependencies.

4. **Interpretability**: Attention weights can be visualized to understand which variables the model focuses on for predictions.

## Training

### Single-Stage Training

```bash
# Train NXRO-Transformer (single-stage)
python NXRO_train_out_of_sample.py --model transformer --epochs 1500 --device auto
```

### Two-Stage Training

```bash
# Train NXRO-Transformer (two-stage: synthetic pre-training → ORAS5 fine-tuning)
python NXRO_train_out_of_sample.py --model transformer --epochs 1500 --device auto \
    --two_stage --extra_train_nc auto
```

### Training Details

- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 1e-3
- **Batch Size**: 128
- **Gradient Clipping**: Max norm 1.0
- **Loss Function**: MSE on one-step-ahead predictions
- **Rollout**: Supports k-step rollout for multi-step training

## Comparison with Other Models

### vs. NXRO-Neural (Pure MLP)
- **Transformer**: Uses self-attention to model variable interactions
- **Neural**: Uses masked MLPs with fixed connectivity

### vs. NXRO-Attentive (Linear + Attention)
- **Transformer**: Pure attention-based, no explicit linear operator
- **Attentive**: Combines seasonal linear operator with attention residual

### vs. NXRO-Graph (Linear + GNN)
- **Transformer**: Learns interactions via attention (dense, state-dependent)
- **Graph**: Uses fixed or learned sparse graph structure

## Implementation Files

- **Model Definition**: `nxro/models.py` → `NXROTransformerModel`
- **Training Function**: `nxro/train.py` → `train_nxro_transformer`
- **Single-Stage Runner**: `run_utils.py` → `run_transformer`
- **Two-Stage Runner**: `run_utils_twostage.py` → `run_transformer_twostage`
- **Main Script**: `NXRO_train_out_of_sample.py` (with `--model transformer`)

## Expected Performance

Based on similar architectures in the NXRO family:

- **Training RMSE**: ~0.47-0.52°C (in-sample)
- **Test RMSE**: ~0.57-0.60°C (out-of-sample)
- **Test ACC**: ~0.60-0.62 (out-of-sample)

Performance may improve with:
- Hyperparameter tuning (d_model, nhead, num_layers)
- Two-stage training (pre-training on synthetic data)
- Longer training (more epochs)
- Data augmentation (multiple synthetic datasets)

## Future Enhancements

1. **Temporal Attention**: Add attention over time steps, not just variables
2. **Hierarchical Structure**: Multi-scale attention for different timescales
3. **Sparse Attention**: Use sparse attention patterns to reduce computation
4. **Cross-Attention**: Incorporate external forcing (e.g., solar, volcanic)
5. **Ensemble**: Combine multiple Transformer models with different initializations

## References

- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.
- Zhou et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI.

