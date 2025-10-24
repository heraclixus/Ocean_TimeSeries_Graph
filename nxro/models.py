import math
from typing import Optional

import torch
import torch.nn as nn
from typing import Optional


def fourier_time_embedding(t: torch.Tensor, k_max: int = 2) -> torch.Tensor:
    """Build seasonal Fourier features [1, cos(ωt), sin(ωt), ..., cos(kωt), sin(kωt)].

    Args:
        t: shape [B] time in years (float, e.g., 1979.0 + m/12)
        k_max: maximum harmonic (K=2 to mirror XRO ac_order=2)
    Returns:
        emb: shape [B, 1 + 2*k_max]
    """
    # ω = 2π (per year), t already in years
    omega = 2.0 * math.pi
    feats = [torch.ones_like(t)]
    for k in range(1, k_max + 1):
        feats.append(torch.cos(k * omega * t))
        feats.append(torch.sin(k * omega * t))
    return torch.stack(feats, dim=-1)


class NXROLinearModel(nn.Module):
    """NXRO-Linear: Seasonal linear operator with Fourier time embedding.

    dX/dt = L_θ(t) · X, where L_θ(t) = sum_k W_k * φ_k(t), φ_k are Fourier features.
    """

    def __init__(self, n_vars: int, k_max: int = 2):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        # Number of seasonal basis terms: 1 + 2*k_max
        self.n_basis = 1 + 2 * k_max
        # Parameterize L as a stack of basis-weighted matrices: [n_basis, n_vars, n_vars]
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        # Initialize small to be close to zero drift initially
        nn.init.xavier_uniform_(self.L_basis)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute f(x,t) = dX/dt.

        Args:
            x: [B, n_vars]
            t_years: [B] absolute time in years (e.g., 1979 + m/12). Used for seasonal embedding
        Returns:
            dxdt: [B, n_vars]
        """
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        # Combine basis to form L(t): [B, n_vars, n_vars]
        # Weighted sum over basis: sum_b emb[b]*L_basis[b]
        # Do batch matmul via einsum: [B, n_basis] * [n_basis, n_vars, n_vars] -> [B, n_vars, n_vars]
        # Combine basis without adding a batch dim to L_basis: [B,K] x [K,U,V] -> [B,U,V]
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        # dx/dt = L(t) @ x
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        return dxdt


class NXROROModel(nn.Module):
    """NXRO-RO: Seasonal linear operator + RO nonlinear terms for T/H.

    dX/dt = L_θ(t) · X + [RO_T(T,H,t), RO_H(T,H,t), 0, ..., 0]^T

    RO monomials: [T^2, T*H, T^3, T^2*H, T*H^2] with seasonal coefficients.
    """

    def __init__(self, n_vars: int, k_max: int = 2, n_ro: int = 5):
        super().__init__()
        assert n_vars >= 2, "NXROROModel assumes first two vars are T and H"
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_ro = n_ro
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # Seasonal RO coefficients for T and H tendencies
        self.W_T = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        self.W_H = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        nn.init.xavier_uniform_(self.W_T)
        nn.init.xavier_uniform_(self.W_H)

    def _phi_ro(self, T: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # Build RO basis [T^2, T*H, T^3, T^2*H, T*H^2]
        return torch.stack([
            T * T,
            T * H,
            T * T * T,
            T * T * H,
            T * H * H,
        ], dim=-1)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        # Linear part
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)  # [B, U, V]
        dxdt = torch.einsum('buv,bv->bu', L_t, x)  # [B, U]
        # RO part on first two variables
        T = x[:, 0]
        H = x[:, 1]
        phi = self._phi_ro(T, H)  # [B, n_ro]
        cT = torch.einsum('bk,kn->bn', emb, self.W_T)  # [B, n_ro]
        cH = torch.einsum('bk,kn->bn', emb, self.W_H)  # [B, n_ro]
        ro_T = torch.sum(cT * phi, dim=-1)  # [B]
        ro_H = torch.sum(cH * phi, dim=-1)  # [B]
        dxdt[:, 0] = dxdt[:, 0] + ro_T
        dxdt[:, 1] = dxdt[:, 1] + ro_H
        return dxdt


class NXRORODiagModel(nn.Module):
    """NXRO-RO+Diag: Seasonal linear + RO (T/H) + diagonal quadratic/cubic per variable.

    dX/dt = L_θ(t) · X + RO_T/H(T,H,t) on first two vars + b_j(t) X_j^2 + c_j(t) X_j^3.
    """

    def __init__(self, n_vars: int, k_max: int = 2, n_ro: int = 5):
        super().__init__()
        assert n_vars >= 2, "NXRORODiagModel assumes first two vars are T and H"
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_ro = n_ro
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # Seasonal RO coefficients for T and H tendencies
        self.W_T = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        self.W_H = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        nn.init.xavier_uniform_(self.W_T)
        nn.init.xavier_uniform_(self.W_H)
        # Seasonal diagonal quadratic and cubic: [basis, n_vars]
        self.B_diag = nn.Parameter(torch.zeros(self.n_basis, n_vars))
        self.C_diag = nn.Parameter(torch.zeros(self.n_basis, n_vars))
        nn.init.xavier_uniform_(self.B_diag)
        nn.init.xavier_uniform_(self.C_diag)

    def _phi_ro(self, T: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            T * T,
            T * H,
            T * T * T,
            T * T * H,
            T * H * H,
        ], dim=-1)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        # Linear part
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        # RO part (T/H)
        T = x[:, 0]
        H = x[:, 1]
        phi = self._phi_ro(T, H)  # [B, n_ro]
        cT = torch.einsum('bk,kn->bn', emb, self.W_T)
        cH = torch.einsum('bk,kn->bn', emb, self.W_H)
        dxdt[:, 0] = dxdt[:, 0] + torch.sum(cT * phi, dim=-1)
        dxdt[:, 1] = dxdt[:, 1] + torch.sum(cH * phi, dim=-1)
        # Diagonal quadratic/cubic
        b_t = torch.einsum('bk,kv->bv', emb, self.B_diag)  # [B, n_vars]
        c_t = torch.einsum('bk,kv->bv', emb, self.C_diag)  # [B, n_vars]
        dxdt = dxdt + b_t * (x * x) + c_t * (x * x * x)
        return dxdt


class NXROResModel(nn.Module):
    """NXRO-Res: Seasonal linear + optional RO/Diag (set to minimal) + residual MLP R_θ(X,t).

    dX/dt = L_θ(t) · X + R_θ([X, φ(t)]) where φ(t) are Fourier features. Residual is small (regularized in training).
    """

    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # Residual MLP: inputs X (n_vars) + time emb (n_basis)
        in_dim = n_vars + self.n_basis
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_vars),
        )

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt_lin = torch.einsum('buv,bv->bu', L_t, x)
        res_in = torch.cat([x, emb], dim=-1)
        dxdt_res = self.residual(res_in)
        return dxdt_lin + dxdt_res


class NXROResidualMixModel(NXRORODiagModel):
    """NXRO-ResidualMix: RO+Diag base plus a small residual MLP scaled by α.

    f(x,t) = f_RO+Diag(x,t) + α · R_θ([x, φ(t)]), with α small (fixed or learnable).
    """

    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64,
                 alpha_init: float = 0.1, alpha_learnable: bool = False, alpha_max: float = 0.5,
                 dropout: float = 0.0):
        super().__init__(n_vars=n_vars, k_max=k_max, n_ro=5)
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.alpha_learnable = alpha_learnable
        self.alpha_max = float(alpha_max)

        in_dim = n_vars + self.n_basis
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_vars)]
        self.residual = nn.Sequential(*layers)

        if alpha_learnable:
            # Parameterize α = α_max * sigmoid(a); init so α≈alpha_init
            eps = 1e-6
            init_ratio = max(min(alpha_init / max(alpha_max, eps), 1 - eps), eps)
            init_logit = math.log(init_ratio / (1.0 - init_ratio))
            self.alpha_param = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer('alpha_const', torch.tensor(float(alpha_init), dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        if self.alpha_learnable:
            return self.alpha_max * torch.sigmoid(self.alpha_param)
        else:
            return self.alpha_const

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        base = super().forward(x, t_years)
        emb = fourier_time_embedding(t_years, self.k_max)
        res_in = torch.cat([x, emb], dim=-1)
        dx_res = self.residual(res_in)
        return base + self.alpha() * dx_res

class NXRONeuralODEModel(nn.Module):
    """NXRO-NeuralODE: Seasonal linear + general drift MLP with optional structural masks.

    dX/dt = L_θ(t) · X + G_θ([X, φ(t)]) + M_masked · X (optional limited cross-variable mixing)

    Args:
        n_vars: number of variables
        k_max: seasonal harmonics
        hidden: MLP hidden size
        depth: number of hidden layers (2-3 recommended for small data)
        dropout: dropout rate inside MLP
        allow_cross: if True, add a masked linear mixing term on X
        mask_mode: 'th_only' (default) allows cross only via T/H columns plus diag; 'full' allows all
    """

    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64, depth: int = 2,
                 dropout: float = 0.0, allow_cross: bool = False, mask_mode: str = 'th_only'):
        super().__init__()
        assert depth >= 1
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.allow_cross = allow_cross
        self.mask_mode = mask_mode

        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)

        # Drift MLP
        in_dim = n_vars + self.n_basis
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            layers += [nn.Tanh()]
        layers += [nn.Linear(hidden, n_vars)]
        self.drift = nn.Sequential(*layers)

        if allow_cross:
            self.W_mix = nn.Parameter(torch.zeros(n_vars, n_vars))
            nn.init.xavier_uniform_(self.W_mix)
            # Build static mask
            if mask_mode == 'th_only':
                mask = torch.zeros(n_vars, n_vars)
                mask[:, 0] = 1.0
                mask[:, 1] = 1.0
                mask += torch.eye(n_vars)
            else:  # 'full'
                mask = torch.ones(n_vars, n_vars)
            self.register_buffer('mix_mask', mask)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, K]
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        drift_in = torch.cat([x, emb], dim=-1)
        dxdt = dxdt + self.drift(drift_in)
        if self.allow_cross:
            dxdt = dxdt + torch.matmul(x, (self.W_mix * self.mix_mask).T)
        return dxdt


class NXROBilinearModel(nn.Module):
    """NXRO-Bilinear: Seasonal linear + low-rank bilinear channels gated by season.

    dX/dt = L_θ(t) · X + (α(t) ⊙ s(X)) · W_proj

    where s_k(X) = sum_{r} (P_k^T X)_r (Q_k^T X)_r with rank r and channels k.
    α_k(t) = φ(t)^T w_k are seasonal gates. W_proj maps channel scalars back to state.
    """

    def __init__(self, n_vars: int, k_max: int = 2, n_channels: int = 2, rank: int = 2):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_channels = n_channels
        self.rank = rank
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # Low-rank bilinear factors per channel
        self.P = nn.Parameter(torch.zeros(n_channels, n_vars, rank))
        self.Q = nn.Parameter(torch.zeros(n_channels, n_vars, rank))
        nn.init.xavier_uniform_(self.P)
        nn.init.xavier_uniform_(self.Q)
        # Seasonal gates per channel
        self.W_alpha = nn.Parameter(torch.zeros(self.n_basis, n_channels))
        nn.init.xavier_uniform_(self.W_alpha)
        # Projection back to state
        self.W_proj = nn.Parameter(torch.zeros(n_channels, n_vars))
        nn.init.xavier_uniform_(self.W_proj)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, K]
        # Linear part
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        # Bilinear scalars per channel
        # p_feat: [B, C, R] where p_feat[:,k,:] = P_k^T x
        p_feat = torch.einsum('bn,knr->bkr', x, self.P)
        q_feat = torch.einsum('bn,knr->bkr', x, self.Q)
        s = torch.sum(p_feat * q_feat, dim=-1)  # [B, C]
        alpha = torch.einsum('bk,kc->bc', emb, self.W_alpha)  # [B, C]
        chan = alpha * s  # [B, C]
        dx_bilin = torch.einsum('bc,cv->bv', chan, self.W_proj)  # [B, V]
        return dxdt + dx_bilin


class NXROAttentiveModel(nn.Module):
    """NXRO-AttentiveCoupling: Seasonal linear + lightweight attention across variables.

    Treat variables as tokens. Single-head attention by default with optional mask emphasizing T/H.
    dx/dt = L(t) x + α(t) * Proj(Softmax(M ⊙ (QK^T / sqrt(d))) V)

    Args:
        n_vars: number of variables
        k_max: seasonal harmonics
        d: attention hidden size (keep small, e.g., 16–32)
        dropout: dropout on attention weights/output
        mask_mode: 'th_only' to allow attention mainly from T/H (cols 0/1) plus self; 'full' for all-to-all
    """

    def __init__(self, n_vars: int, k_max: int = 2, d: int = 32, dropout: float = 0.0, mask_mode: str = 'th_only'):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.d = d
        self.mask_mode = mask_mode
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # Attention projections
        self.Wq = nn.Linear(1, d, bias=False)
        self.Wk = nn.Linear(1, d, bias=False)
        self.Wv = nn.Linear(1, d, bias=False)
        self.Wo = nn.Linear(d, 1, bias=False)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Seasonal gate α(t)
        self.alpha_w = nn.Parameter(torch.zeros(self.n_basis))
        nn.init.normal_(self.alpha_w, std=0.1)

        # Build static attention mask
        mask = torch.zeros(n_vars, n_vars)
        mask += torch.eye(n_vars)
        if mask_mode == 'th_only':
            mask[:, 0] = 1.0
            mask[:, 1] = 1.0
        else:  # 'full'
            mask = torch.ones(n_vars, n_vars)
        self.register_buffer('attn_mask', mask)  # 1=allowed, 0=blocked

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, K]
        # Linear part
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        # Prepare token features: reshape x to [B, V, 1]
        x_tok = x.unsqueeze(-1)
        Q = self.Wq(x_tok)  # [B, V, d]
        K = self.Wk(x_tok)  # [B, V, d]
        Vv = self.Wv(x_tok)  # [B, V, d]
        scale = 1.0 / (self.d ** 0.5)
        scores = torch.einsum('bvd,bud->bvu', Q, K) * scale  # [B, V, V]
        # Apply mask: disallow where mask==0 by large negative
        scores = scores.masked_fill(self.attn_mask.unsqueeze(0) < 0.5, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        O = torch.einsum('bvu,bud->bvd', attn, Vv)  # [B, V, d]
        out = self.Wo(O).squeeze(-1)  # [B, V]
        out = self.out_drop(out)
        # Seasonal gate α(t) in [0,1]
        alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, self.alpha_w))  # [B]
        dxdt = dxdt + out * alpha.unsqueeze(-1)
        return dxdt


class NXROGraphModel(nn.Module):
    """NXRO-GraphNeuralODE: Seasonal linear + sparse graph drift.

    dX/dt = L(t) X + α(t) · tanh( (Â X) W_g ), where Â is (row-)normalized adjacency.

    If use_fixed_graph=True, Â is provided/constructed and frozen; otherwise A is learned with L1 sparsity (handled in training).
    """

    def __init__(self, n_vars: int, k_max: int = 2, use_fixed_graph: bool = True,
                 adj_init: torch.Tensor = None, top_k: int = 2, hidden: int = 0):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.use_fixed_graph = use_fixed_graph
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # Graph parameters
        if use_fixed_graph:
            if adj_init is None:
                adj_init = torch.eye(n_vars)
            self.register_buffer('A_fixed', adj_init)
        else:
            self.A_param = nn.Parameter(torch.zeros(n_vars, n_vars))
            nn.init.xavier_uniform_(self.A_param)
        # Projection matrix for graph message
        self.W_g = nn.Parameter(torch.zeros(n_vars, n_vars))
        nn.init.xavier_uniform_(self.W_g)
        # Seasonal gate α(t)
        self.alpha_w = nn.Parameter(torch.zeros(self.n_basis))
        nn.init.normal_(self.alpha_w, std=0.1)

    def _normalize_rows(self, A: torch.Tensor) -> torch.Tensor:
        A = A + torch.eye(A.shape[0], device=A.device)  # add self-loops
        rowsum = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return A / rowsum

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, K]
        # Linear part
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        # Adjacency (fixed or learned)
        if self.use_fixed_graph:
            A_hat = self._normalize_rows(self.A_fixed)
        else:
            A_pos = torch.relu(self.A_param)  # enforce nonnegativity
            A_hat = self._normalize_rows(A_pos)
        # Graph drift
        message = torch.matmul(x, A_hat.T)  # [B, V]
        dx_graph = torch.tanh(torch.matmul(message, self.W_g.T))  # [B, V]
        alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, self.alpha_w)).unsqueeze(-1)
        return dxdt + alpha * dx_graph


# --------- PyG-based Graph ODE ---------
def build_edge_index_from_corr(corr: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    """Build undirected edge_index from correlation matrix (absolute), keep top_k per node.

    Args:
        corr: [V, V] tensor
        top_k: neighbors per node
    Returns:
        edge_index: [2, E] long tensor (undirected, includes i->j and j->i)
    """
    V = corr.shape[0]
    with torch.no_grad():
        A = corr.abs().clone()
        A.fill_diagonal_(0.0)
        edges = []
        for i in range(V):
            vals, idx = torch.topk(A[i], k=min(top_k, V - 1))
            for j in idx.tolist():
                edges.append([i, j])
                edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).T
    return edge_index


class NXROGraphPyGModel(nn.Module):
    """PyG-based Graph ODE: Seasonal linear + GNN drift (GCN or GAT), with fixed edge_index.

    Args:
        n_vars: number of variables (nodes)
        k_max: seasonal harmonics
        edge_index: [2, E] LongTensor with undirected edges
        hidden: hidden channels for GNN
        dropout: dropout rate in GNN
        use_gat: if True, use GATConv; else GCNConv
    """

    def __init__(self, n_vars: int, k_max: int, edge_index: torch.Tensor,
                 hidden: int = 16, dropout: float = 0.0, use_gat: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.register_buffer('edge_index', edge_index)
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        nn.init.xavier_uniform_(self.L_basis)
        # PyG layers
        try:
            if use_gat:
                from torch_geometric.nn import GATConv as Conv
            else:
                from torch_geometric.nn import GCNConv as Conv
        except Exception as e:
            raise ImportError("torch_geometric is required for NXROGraphPyGModel. Install torch-geometric.")
        self.conv1 = Conv(1, hidden)
        self.conv2 = Conv(hidden, 1)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Seasonal gate
        self.alpha_w = nn.Parameter(torch.zeros(self.n_basis))
        nn.init.normal_(self.alpha_w, std=0.1)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, K]
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        # Process each sample (small n_vars and batches make this acceptable)
        out_list = []
        for b in range(B):
            xb = x[b].unsqueeze(-1)  # [V, 1]
            h = self.conv1(xb, self.edge_index)
            h = self.act(h)
            h = self.drop(h)
            h = self.conv2(h, self.edge_index)
            out_list.append(h.squeeze(-1))
        graph_out = torch.stack(out_list, dim=0)  # [B, V]
        alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, self.alpha_w)).unsqueeze(-1)
        return dxdt + alpha * graph_out


