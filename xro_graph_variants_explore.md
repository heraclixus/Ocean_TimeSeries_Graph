# NXRO-Graph Variants: Exploration for Improved Performance

## Current Status

**Best Graph Model**: NXRO-Graph (Fixed XRO) - Rank 3 (out-of-sample, all datasets)
- Test ACC: 0.607
- Test RMSE: 0.589 C

**Target**: Outperform NXRO-Linear (Rank 1)
- Test ACC: 0.615
- Test RMSE: 0.586 C

**Gap to close**: +0.008 ACC, -0.003 C RMSE

---

## Exploration Scope

This document focuses on **4 main axes** for systematic exploration:

**IN SCOPE (Current Exploration)**:
1. Graph Structure/Topology (FULL)
2. GNN Architecture - GCN and GAT only
3. Training Strategies (FULL)
4. Graph Hyperparameters (FULL)
7. Enhanced Seasonal Features only
8. Edge Sparsity Regularization only

**FUTURE WORK (Deferred)**:
5. Hybrid Architectures
6. Temporal Graphs and Advanced Innovations
9. Ensemble Strategies
10. Data Augmentation

---

## Axis 1: Graph Structure/Topology

The adjacency matrix defines which variables influence each other. Current default uses XRO-derived coupling.

### A1.1: XRO-Derived Adjacency (Current Default)

**Method**: Extract from XRO's fitted linear operator $\mathbf{L}(t)$

**Construction**:
```
1. Fit XRO on training data -> L_ac(i,j,m) [i,j = variables, m = month]
2. Compute strength: S(i,j) = mean_m |L_ac(i,j,m)|
3. Zero diagonal (no self-loops initially)
4. Symmetrize: A = max(S, S^T)
5. Normalize rows, add self-loops
```

**Pros**:
- Physics-informed (captures XRO's learned teleconnections)
- Interpretable (edges correspond to physical coupling)

**Cons**:
- Limited to XRO's linear approximation
- May miss nonlinear interactions
- Coupling strength based on training period only

**Variants to try**:
- **A1.1a**: Use threshold (e.g., keep edges > 0.1 percentile)
- **A1.1b**: Use different aggregation (max over months vs. mean)
- **A1.1c**: Weight edges by coupling strength (weighted adjacency)
- **A1.1d**: Include lagged coupling (use XRO with time lags)

---

### A1.2: Statistical KNN Graphs

**Method**: Compute pairwise statistical interaction on training data, keep top-k neighbors.

**Available metrics**:

**A1.2a: Pearson Correlation**
```
A(i,j) = |corr(X_i, X_j)| over time
```
**Pros**: Simple, captures linear relationships  
**Cons**: Misses nonlinear interactions, sensitive to outliers

**A1.2b: Spearman Rank Correlation**
```
A(i,j) = |rank_corr(X_i, X_j)| over time
```
**Pros**: Robust to outliers, captures monotonic relationships  
**Cons**: Still limited to monotonic patterns

**A1.2c: Mutual Information**
```
A(i,j) = MI(X_i, X_j) via histogram binning
```
**Pros**: Captures nonlinear dependencies  
**Cons**: Sensitive to binning, needs more data

**A1.2d: Cross-Correlation Maximum**
```
A(i,j) = max_lag |corr(X_i(t+lag), X_j(t))| for lag in [-L, +L]
```
**Pros**: Captures lagged teleconnections (e.g., ENSO-IOD lag)  
**Cons**: Computationally expensive, may overfit to lags

**Hyperparameter**: `top_k` (number of neighbors to keep)
- Try: k = 1, 2, 3, 5, 7, 10
- Trade-off: Small k = sparse (regularized), Large k = dense (expressive)

**Variants to try**:
- **A1.2e**: Combine multiple metrics (e.g., union of Pearson top-3 and MI top-3)
- **A1.2f**: Directed graph (don't symmetrize, allow i->j without j->i)
- **A1.2g**: Multi-scale graph (different k for different variables)

---

### A1.3: Learned Adjacency

**Method**: Parameterize adjacency as trainable, optimize with sparsity penalty.

**Formulation**:
```
A_ij = ReLU(a_ij)  where a_ij is trainable
Loss = RMSE + lambda * sum|A_ij| (L1 penalty)
```

**Pros**:
- Fully adaptive to data
- Can discover unexpected teleconnections

**Cons**:
- High overfitting risk (many parameters)
- Needs strong regularization

**Variants to try**:
- **A1.3a**: Start from XRO prior, allow refinement (warm-start adjacency)
- **A1.3b**: Start from statistical KNN, allow refinement
- **A1.3c**: Learn different adjacencies per season (12 graphs, one per month)
- **A1.3d**: Low-rank adjacency: $A = UV^T$ where U,V are n_vars x rank matrices
- **A1.3e**: Gumbel-softmax for discrete edge selection (differentiable sparsity)

**Hyperparameters**:
- L1 penalty: Try [1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
- Learning rate for adjacency (separate from other params): Try [1e-5, 1e-4]

---

### A1.4: Hybrid Graph Structures

**A1.4a: Hierarchical (Multi-Scale) Graph**
- ENSO core (T, H, Nino34): Dense subgraph (fully connected)
- Other modes: Sparse connections (k-NN)
- Inter-group: Based on physical knowledge (e.g., IOD connects to Nino34)

**A1.4b: Multi-Channel Graph**
- Different adjacency for different types of coupling:
  - Channel 1: Short-term (high correlation, k=3)
  - Channel 2: Long-term (lagged correlation, k=2)
  - Channel 3: Nonlinear (MI, k=2)
- Aggregate: Weighted sum of channels

**A1.4c: Dynamic Graph**
- Adjacency changes with time: $A(t) = A_0 + \Delta A(t)$
- $\Delta A(t) = \text{MLP}(\boldsymbol{\phi}(t))$ where $\boldsymbol{\phi}$ are seasonal features
- Captures seasonal variation in teleconnection strength

**A1.4d: Attention-Weighted Graph**
- Fixed topology from XRO, but edge weights learned via attention:
```
A_effective(i,j) = A_topology(i,j) * attention_weight(X_i, X_j)
```

---

## Axis 2: Graph Neural Network Architecture

### A2.1: Graph Convolution Type [IN SCOPE]

**A2.1a: GCN (Graph Convolutional Network) - Current**
```
H = tanh(A * X * W)
```
**Pros**: Simple, stable  
**Cons**: All neighbors contribute equally (no selectivity)

**A2.1b: GAT (Graph Attention Networks)**
```
H_i = sum_j alpha_ij * W * X_j
where alpha_ij = softmax(LeakyReLU(a^T [W*X_i || W*X_j]))
```
**Pros**: Learns attention weights per edge (state-dependent)  
**Cons**: More parameters, higher overfitting risk

**Note**: GraphSAGE and GIN variants are deferred to future work.

---

### A2.2: Number of Graph Layers [DEFERRED]

**Current**: Single graph conv layer

**Status**: Deferred to future work. Focus on single-layer models with topology and hyperparameter optimization first.

---

### A2.3: Hidden Dimensions [DEFERRED]

**Current**: hidden = 16 (for PyG models)

**Status**: Deferred to future work after topology is optimized. Current hidden=16 is reasonable for single-layer models.

---

### A2.4: Aggregation Functions [DEFERRED]

**Current**: Sum (GCN) or attention-weighted (GAT)

**Status**: Deferred. GCN and GAT already cover main aggregation types (sum vs. attention-weighted).

---

### A2.5: Activation Functions [DEFERRED]

**Current**: tanh

**Status**: Deferred. Tanh works well for bounded climate variables.

---

## Axis 3: Training Strategies

### A3.1: Warm-Start Options

**A3.1a: Random Init (Current for Rank 3 Model)**
- Start from scratch

**A3.1b: Warm-Start Linear Operator**
- Initialize $\mathbf{L}_\theta$ from XRO
- Train all parameters (linear + graph)
- Faster convergence, but may bias toward XRO's solution

**A3.1c: Freeze Linear, Train Graph Only**
- Fix $\mathbf{L}_\theta$ from XRO
- Train only graph convolution weights
- Tests if XRO's linear operator + learned graph beats full training

**A3.1d: Progressive Unfreezing**
```
Phase 1 (epochs 1-100): Freeze linear, train graph
Phase 2 (epochs 101-200): Unfreeze all, fine-tune jointly
```
- Staged training, graph learns first under fixed linear

---

### A3.2: Learning Rate Schedules

**Current**: Constant learning rate (likely 1e-3)

**A3.2a: Cosine Annealing**
```
lr(t) = lr_min + (lr_max - lr_min) * (1 + cos(pi*t/T)) / 2
```
Gradually reduces LR, helps fine-tuning.

**A3.2b: Warm-Up + Decay**
```
Epochs 1-50: Linear warm-up (0 -> lr_max)
Epochs 51+: Exponential decay
```

**A3.2c: Separate LR for Components**
```
Linear operator: 1e-4 (if warm-started)
Graph weights: 1e-3 (randomly initialized)
Adjacency (if learned): 1e-5 (slow adaptation)
```

---

### A3.3: Multi-Step Rollout Training

**Current**: Single-step loss (predict X(t+1) from X(t))

**A3.3a: 2-Step Rollout**
```
Loss = RMSE(X(t+1)) + beta * RMSE(X(t+2))
```
Encourages longer-term consistency.

**A3.3b: Progressive Rollout**
```
Epochs 1-500: 1-step
Epochs 501-1000: 2-step
Epochs 1001+: 3-step
```

**A3.3c: Random Rollout Length**
Randomly sample k from [1, K_max] each batch, improves robustness.

**Recommendation**: Try 2-step rollout with beta=0.5.

---

### A3.4: Regularization Strategies

**A3.4a: Dropout Variants**
- **Node dropout**: Randomly drop variables during training
- **Edge dropout**: Randomly drop graph edges
- **Feature dropout**: Current (dropout on hidden features)

**A3.4b: Weight Decay Tuning**
- Current: Likely 1e-4
- Try: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
- Graph weights may need different decay than linear operator

**A3.4c: Spectral Normalization**
Apply to graph weight matrices to constrain Lipschitz constant:
```
W_normalized = W / max_singular_value(W)
```
Prevents exploding gradients in message passing.

**A3.4d: Early Stopping Variants**
- Current: Stop on validation RMSE
- Try: Stop on validation ACC
- Try: Stop when train-test gap exceeds threshold

---

## Axis 4: Graph Construction Hyperparameters

### A4.1: Top-K Selection

**Current**: k=3 (likely default)

**Grid search**: k in [1, 2, 3, 5, 7, 10]

**Hypothesis**:
- Very sparse (k=1,2): Strong regularization, may miss important connections
- Medium (k=3,5): Balance sparsity and connectivity
- Dense (k=7,10): More expressive, higher overfitting risk

**Data-dependent selection**: Use validation loss to pick k automatically.

---

### A4.2: Edge Weighting Schemes

**Current**: Binary adjacency (edge present = 1, absent = 0)

**A4.2a: Coupling-Strength Weighted**
```
A(i,j) = S(i,j) / sum_k S(i,k)  (row-normalized strength)
```
Strong couplings weighted higher.

**A4.2b: Distance-Based Weighting**
For statistical graphs:
```
A(i,j) = exp(-d(i,j) / sigma)
where d(i,j) = 1 - |corr(X_i, X_j)|
```
Gaussian kernel on correlation distance.

**A4.2c: Learned Edge Weights**
```
A(i,j) = topology(i,j) * w_ij
where topology is fixed (XRO/statistical), w_ij is learned
```

---

### A4.3: Self-Loop Handling

**Current**: Add self-loops with weight 1.0

**A4.3a: No Self-Loops**
Force pure message passing from neighbors.

**A4.3b: Learned Self-Loop Weight**
```
H_i = w_self * X_i + sum_j A_ij * message_j
where w_self is trainable
```

**A4.3c: Variable-Specific Self-Loop Weights**
Different w_self for each variable (some variables more autocorrelated).

---

## Axis 5: Hybrid Architectures [DEFERRED TO FUTURE WORK]

Hybrid architectures (Graph+MLP, Graph+RO, etc.) are deferred. Focus on optimizing standalone graph models first.

---

## Axis 6: Architectural Innovations [DEFERRED TO FUTURE WORK]

Advanced innovations including:
- Temporal graphs (season-specific adjacency)
- Directed graphs (causal structure)
- Multi-hop message passing
- Virtual nodes

These are kept in scope for future exploration after baseline graph optimization is complete.

---

## Axis 7: Input Features [IN SCOPE: Enhanced Seasonal Only]

### A7.1: Enhanced Seasonal Features [IN SCOPE]

**Current**: $\boldsymbol{\phi}(t) = [1, \cos(\omega t), \sin(\omega t), \cos(2\omega t), \sin(2\omega t)]^T$ (k_max=2)

**A7.1a: Extended Harmonics (k_max=3)**
Add k=3 terms:
```
phi(t) = [1, cos(t), sin(t), cos(2t), sin(2t), cos(3t), sin(3t)]
```
**Rationale**: Capture higher-frequency seasonal variations (tri-annual cycles).

**A7.1b: Extended Harmonics (k_max=4)**
Add k=3,4 terms:
```
phi(t) = [1, cos(t), sin(t), cos(2t), sin(2t), cos(3t), sin(3t), cos(4t), sin(4t)]
```
**Rationale**: Maximum seasonal detail, but risk overfitting.

**A7.1c: Reduced Harmonics (k_max=1)**
Only annual cycle:
```
phi(t) = [1, cos(t), sin(t)]
```
**Rationale**: Extreme regularization, test if semi-annual terms are necessary.

**Experiments**: Test k_max in [1, 2, 3, 4]

---

### A7.2: Other Feature Engineering [DEFERRED]

Node features (normalized, augmented, PCA), month embeddings, and phase embeddings are deferred to future work.

---

## Axis 8: Regularization [IN SCOPE: Sparsity Only]

### A8.1: Edge Sparsity Penalty [IN SCOPE]

**A8.1a: L1 Penalty on Adjacency (Learned Graphs Only)**
```
Loss = RMSE + lambda_sparse * |A|_1
```
Encourages sparse graphs (fewer edges).

**Hyperparameters to try**: lambda_sparse in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

**Trade-off**:
- Small lambda: Dense graph, high capacity, overfitting risk
- Large lambda: Sparse graph, regularized, may miss connections

**Experiments**:
```
for lambda in [1e-5, 1e-4, 1e-3]:
  python NXRO_train_out_of_sample.py --model graph \
    --graph_learned --graph_l1 $lambda --eval_all_datasets
```

---

### A8.2: Weight Decay [IN SCOPE]

**Current**: Likely 1e-4 (default)

**Experiments**: Try [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

**Rationale**: Graph weights may need different regularization than linear operator.

---

### A8.3: Dropout Tuning [IN SCOPE]

**Current**: dropout = 0.1

**Experiments**: Try [0.0, 0.1, 0.2, 0.3]

**Rationale**: Higher dropout may prevent overfitting on graph features.

---

### A8.4: Other Regularization [DEFERRED]

Smoothness penalties, Laplacian regularization, spectral normalization are deferred to future work.

---

## Axis 9: Ensemble Strategies [DEFERRED TO FUTURE WORK]

Ensemble methods (graph ensembles, stacking, Graph+Linear combinations) are deferred. Focus on single-model optimization first.

---

## Axis 10: Data Augmentation [DEFERRED TO FUTURE WORK]

Data augmentation techniques (synthetic nodes, temporal augmentation, multi-dataset training) are deferred to future work.

---

## Recommended Experimental Plan (Focused)

### Phase 1: Graph Topology Search [HIGH PRIORITY]

**Objective**: Find best graph structure with fixed GCN/GAT architecture.

**Experiments** (25 runs):
```bash
# XRO-derived with different sparsity
for k in 1 2 3 5 7; do
  python NXRO_train_out_of_sample.py --model graph_pyg --top_k $k --eval_all_datasets --epochs 1500
  python NXRO_train_out_of_sample.py --model graph_pyg --gat --top_k $k --eval_all_datasets --epochs 1500
done

# Statistical priors with k=3 (baseline comparison)
for method in pearson spearman mi xcorr_max; do
  python NXRO_train_out_of_sample.py --model graph_pyg \
    --graph_stat_method $method --graph_stat_topk 3 --eval_all_datasets --epochs 1500
  
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $method --graph_stat_topk 3 --eval_all_datasets --epochs 1500
done

# Best statistical prior with varied sparsity
BEST_METHOD=<from above, likely mi or xcorr_max>
for k in 2 5 7; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_METHOD --graph_stat_topk $k --eval_all_datasets --epochs 1500
done
```

**Expected outcome**: Identify best (topology, k, GNN_type) combination.

**Time**: ~25-30 hours on GPU (parallelizable across GPUs)

---

### Phase 2: Training Strategy Optimization [HIGH PRIORITY]

**Objective**: Optimize training for best topology from Phase 1.

**Fix**: BEST_TOPOLOGY, BEST_K, BEST_GNN from Phase 1

**Experiments** (18 runs):

**2.1: Warm-Start Variants (3 runs)**
```bash
# Random init (baseline)
python NXRO_train_out_of_sample.py --model graph_pyg --gat \
  --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K --eval_all_datasets

# Warm-start linear, train all
python NXRO_train_out_of_sample.py --model graph_pyg --gat \
  --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
  --warm_start results_all_outsample/xro_fit_warmstart.nc --eval_all_datasets

# Freeze linear, train graph only
python NXRO_train_out_of_sample.py --model graph_pyg --gat \
  --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
  --warm_start results_all_outsample/xro_fit_warmstart.nc --freeze linear --eval_all_datasets
```

**2.2: Learning Rate Tuning (5 runs)**
```bash
for lr in 5e-4 1e-3 2e-3 5e-3 1e-2; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --lr $lr --eval_all_datasets --epochs 1500
done
```

**2.3: Multi-Step Rollout (4 runs)**
```bash
for rollout_k in 1 2 3 5; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --rollout_k $rollout_k --eval_all_datasets --epochs 1500
done
```

**2.4: Epoch Count (3 runs)**
```bash
for epochs in 1000 1500 2000; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --epochs $epochs --eval_all_datasets
done
```

**2.5: Batch Size (3 runs)**
```bash
for batch in 64 128 256; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --batch_size $batch --eval_all_datasets --epochs 1500
done
```

**Time**: ~18-20 hours on GPU

---

### Phase 3: Regularization Tuning [MEDIUM PRIORITY]

**Objective**: Fine-tune regularization for best model from Phase 2.

**Fix**: BEST_TOPOLOGY, BEST_K, BEST_GNN, BEST_LR from Phase 2

**Experiments** (12 runs):

**3.1: Weight Decay Grid (5 runs)**
```bash
for wd in 1e-5 5e-5 1e-4 5e-4 1e-3; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --lr $BEST_LR --weight_decay $wd --eval_all_datasets --epochs 1500
done
```

**3.2: Dropout Grid (4 runs)**
```bash
for dropout in 0.0 0.1 0.2 0.3; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --lr $BEST_LR --dropout $dropout --eval_all_datasets --epochs 1500
done
```

**3.3: Learned Adjacency with Sparsity (3 runs)**
```bash
for l1 in 1e-4 5e-4 1e-3; do
  python NXRO_train_out_of_sample.py --model graph \
    --graph_learned --graph_l1 $l1 \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --eval_all_datasets --epochs 1500
done
```

**Time**: ~12-15 hours on GPU

---

### Phase 4: Seasonal Feature Exploration [LOW PRIORITY]

**Objective**: Test if extended harmonics help.

**Fix**: Best configuration from Phase 3

**Experiments** (4 runs):

**Note**: Requires modifying code to accept k_max parameter.

```bash
for k_max in 1 2 3 4; do
  python NXRO_train_out_of_sample.py --model graph_pyg --gat \
    --graph_stat_method $BEST_TOPOLOGY --graph_stat_topk $BEST_K \
    --k_max $k_max --eval_all_datasets --epochs 1500
done
```

**Time**: ~4-6 hours on GPU

---

## Expected Performance Targets

### Conservative Target (Achievable)

Match or slightly beat NXRO-Linear:
- Test ACC: 0.615-0.620
- Test RMSE: 0.580-0.586 C

**Strategy**: Phase 1 (topology search) alone may achieve this.

---

### Target Success

Beat NXRO-Linear convincingly:
- Test ACC: 0.620-0.625
- Test RMSE: 0.575-0.580 C

**Strategy**: Phases 1-3 (topology + training + regularization)

---

### Stretch Goal

Establish clear superiority:
- Test ACC: 0.630+
- Test RMSE: 0.570 C

**Strategy**: All phases + future work (hybrids, temporal graphs)

---

## Implementation Priorities (Focused Plan)

### Immediate Execution (All Ready to Run)

**Priority 1: Graph Topology Search** (Phase 1)
- 25 experiments
- Already fully implemented
- Expected largest impact
- Estimated time: 25-30 hours GPU

**Priority 2: Training Strategy Optimization** (Phase 2)
- 18 experiments
- Already fully implemented
- Critical for convergence
- Estimated time: 18-20 hours GPU

**Priority 3: Regularization Tuning** (Phase 3)
- 12 experiments
- Already fully implemented
- May provide 1-2% improvement
- Estimated time: 12-15 hours GPU

**Priority 4: Seasonal Features** (Phase 4)
- 4 experiments
- Requires minor code modification (expose k_max parameter)
- Marginal gains expected
- Estimated time: 4-6 hours GPU

**Total**: ~60-70 hours GPU time, ~59 experiments

---

### Deferred to Future Work (Not in Current Plan)

- Hybrid architectures (Graph+MLP, Graph+RO)
- Multi-layer GNN (>1 layer)
- Temporal graphs (seasonal adjacency)
- Advanced message passing
- Ensemble methods
- Data augmentation
- Hidden dimension search (fixed at 16 for now)
- Aggregation function variants
- Activation function variants

---

## Quick Start: Run Phase 1 Now

Already implemented and ready to run:

```bash
# XRO-derived topology with different k
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 1 --eval_all_datasets
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 2 --eval_all_datasets
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 3 --eval_all_datasets
python NXRO_train_out_of_sample.py --model graph_pyg --top_k 5 --eval_all_datasets

# Statistical topologies
python NXRO_train_out_of_sample.py --model graph_pyg --gat \
  --graph_stat_method pearson --graph_stat_topk 3 --eval_all_datasets

python NXRO_train_out_of_sample.py --model graph_pyg --gat \
  --graph_stat_method mi --graph_stat_topk 3 --eval_all_datasets

python NXRO_train_out_of_sample.py --model graph_pyg --gat \
  --graph_stat_method xcorr_max --graph_stat_topk 3 --eval_all_datasets

# GAT vs GCN
python NXRO_train_out_of_sample.py --model graph_pyg --gat --eval_all_datasets
python NXRO_train_out_of_sample.py --model graph_pyg --eval_all_datasets  # GCN

# Learned adjacency
python NXRO_train_out_of_sample.py --model graph \
  --graph_learned --graph_l1 1e-4 --eval_all_datasets
```

After Phase 1, run ranking to identify best configuration:
```bash
python rank_all_variants_outsample.py --top_n 20 --metric rmse --force
```

Then proceed to Phase 2-4 with best topology.

---

## Success Criteria

### Minimum Success

**Graph variant in top 3** (currently #3, already achieved)

### Target Success

**Graph variant ranks #1**, beating NXRO-Linear

### Stretch Success

**Graph variant beats NXRO-Linear by >1% ACC or >0.01 C RMSE**, establishing graph structure as clearly superior for climate forecasting.

---

## Analysis Plan

After running experiments:

1. **Topology comparison**: Which graph prior generalizes best?
2. **Sparsity analysis**: Plot performance vs. k (sparsity-accuracy trade-off)
3. **Attention visualization**: If GAT wins, visualize learned attention weights
4. **Ablation study**: Graph contribution = Performance - Linear_only_performance
5. **Cross-dataset robustness**: Which topology is most consistent across datasets?

---

## Code Modifications Needed

### Immediate (For Phase 1-2, Already Mostly Implemented)

None! Use existing scripts with different flags.

### Near-Term (For Phase 3)

1. **Graph + MLP Hybrid**: New model class `NXROGraphResModel`
2. **Graph + RO**: New model class `NXROGraphROModel`

### Long-Term (For Advanced Techniques)

3. **Multi-layer GNN**: Extend `NXROGraphPyGModel` to support `num_layers` parameter
4. **Temporal graphs**: New model class `NXROTemporalGraphModel`
5. **Progressive unfreezing**: Add to training loop logic

---

## References

- **Graph construction**: See `graph_construction.py`
- **Current implementation**: See `nxro/models.py` (NXROGraphModel, NXROGraphPyGModel)
- **Training**: See `nxro/train.py` (train_nxro_graph, train_nxro_graph_pyg)
- **Baseline XRO**: Zhao et al. (2024), Nature

---

## Expected Timeline (Focused Plan)

- **Phase 1** (topology search): 25-30 GPU hours (~1-2 days with single GPU, <1 day parallelized)
- **Phase 2** (training optimization): 18-20 GPU hours (~1 day)
- **Phase 3** (regularization): 12-15 GPU hours (~0.5-1 day)
- **Phase 4** (seasonal features): 4-6 GPU hours (~0.5 day, requires minor code change)

**Total**: ~60-70 GPU hours, ~3-5 days sequential execution, ~1-2 days if parallelized across 3-4 GPUs.

**Recommendation**: Execute Phase 1-3 sequentially. Each phase informs the next.

