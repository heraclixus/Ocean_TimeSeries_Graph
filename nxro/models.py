import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    
    Variants:
        - Variant 1 (NXRO-Linear): Random initialization (L_basis_init=None)
        - Variant 1a (NXRO-Linear-WS): Warm-start from XRO (L_basis_init from XRO fit)
    """

    def __init__(self, n_vars: int, k_max: int = 2, L_basis_init: Optional[torch.Tensor] = None):
        """Initialize NXRO-Linear model.
        
        Args:
            n_vars: number of variables
            k_max: maximum harmonic order (default 2 for semi-annual)
            L_basis_init: optional initial values for L_basis [n_basis, n_vars, n_vars]
                         If None: random Xavier initialization (Variant 1)
                         If provided: warm-start from XRO (Variant 1a)
        """
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        # Number of seasonal basis terms: 1 + 2*k_max
        self.n_basis = 1 + 2 * k_max
        # Parameterize L as a stack of basis-weighted matrices: [n_basis, n_vars, n_vars]
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        
        # Initialize L_basis
        if L_basis_init is not None:
            # Variant 1a: Warm-start from XRO
            assert L_basis_init.shape == (self.n_basis, n_vars, n_vars), \
                f"L_basis_init shape {L_basis_init.shape} must match ({self.n_basis}, {n_vars}, {n_vars})"
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            # Variant 1: Random initialization
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
    
    Variants:
        - Variant 2: Random initialization (all init params = None, all freeze flags = False)
        - Variant 2a (NXRO-RO-WS): Warm-start (provide init params, freeze_linear=False, freeze_ro=False)
        - Variant 2a-FixL: Warm-start with frozen linear (freeze_linear=True, freeze_ro=False)
        - Variant 2a-FixRO: Warm-start with frozen RO (freeze_linear=False, freeze_ro=True)
        - Variant 2a-FixAll: All frozen, no training (freeze_linear=True, freeze_ro=True)
    """

    def __init__(self, n_vars: int, k_max: int = 2, n_ro: int = 5,
                 L_basis_init: Optional[torch.Tensor] = None,
                 W_T_init: Optional[torch.Tensor] = None,
                 W_H_init: Optional[torch.Tensor] = None,
                 freeze_linear: bool = False,
                 freeze_ro: bool = False):
        """Initialize NXRO-RO model.
        
        Args:
            n_vars: number of variables
            k_max: maximum harmonic order
            n_ro: number of RO basis functions (default 5)
            L_basis_init: optional initial values for L_basis [n_basis, n_vars, n_vars]
            W_T_init: optional initial values for W_T [n_basis, n_ro]
            W_H_init: optional initial values for W_H [n_basis, n_ro]
            freeze_linear: if True, freeze L_basis (requires L_basis_init)
            freeze_ro: if True, freeze W_T and W_H (requires W_T_init, W_H_init)
        """
        super().__init__()
        assert n_vars >= 2, "NXROROModel assumes first two vars are T and H"
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_ro = n_ro
        
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis)
        
        # Freeze linear if requested
        if freeze_linear:
            self.L_basis.requires_grad = False
        
        # Seasonal RO coefficients for T and H tendencies
        self.W_T = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        self.W_H = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        
        if W_T_init is not None and W_H_init is not None:
            with torch.no_grad():
                self.W_T.copy_(W_T_init)
                self.W_H.copy_(W_H_init)
        else:
            nn.init.xavier_uniform_(self.W_T)
            nn.init.xavier_uniform_(self.W_H)
        
        # Freeze RO if requested
        if freeze_ro:
            self.W_T.requires_grad = False
            self.W_H.requires_grad = False

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
    
    Variants:
        - Variant 3: Random initialization
        - Variant 3a (NXRO-RO+Diag-WS): Warm-start all, train all
        - Variant 3a-FixL: Warm-start, freeze linear, train RO+Diag
        - Variant 3a-FixRO: Warm-start, freeze RO, train linear+Diag
        - Variant 3a-FixDiag: Warm-start, freeze diagonal, train linear+RO
        - Variant 3a-FixNL: Warm-start, freeze RO+Diag, train linear only
        - Variant 3a-FixAll: Warm-start, freeze all (pure XRO baseline)
    """

    def __init__(self, n_vars: int, k_max: int = 2, n_ro: int = 5,
                 L_basis_init: Optional[torch.Tensor] = None,
                 W_T_init: Optional[torch.Tensor] = None,
                 W_H_init: Optional[torch.Tensor] = None,
                 B_diag_init: Optional[torch.Tensor] = None,
                 C_diag_init: Optional[torch.Tensor] = None,
                 freeze_linear: bool = False,
                 freeze_ro: bool = False,
                 freeze_diag: bool = False):
        """Initialize NXRO-RO+Diag model.
        
        Args:
            n_vars: number of variables
            k_max: maximum harmonic order
            n_ro: number of RO basis functions
            L_basis_init: optional initial values for L_basis
            W_T_init, W_H_init: optional initial values for RO coefficients
            B_diag_init, C_diag_init: optional initial values for diagonal coefficients
            freeze_linear: if True, freeze L_basis
            freeze_ro: if True, freeze W_T and W_H
            freeze_diag: if True, freeze B_diag and C_diag
        """
        super().__init__()
        assert n_vars >= 2, "NXRORODiagModel assumes first two vars are T and H"
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_ro = n_ro
        
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis)
        if freeze_linear:
            self.L_basis.requires_grad = False
        
        # Seasonal RO coefficients for T and H tendencies
        self.W_T = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        self.W_H = nn.Parameter(torch.zeros(self.n_basis, n_ro))
        if W_T_init is not None and W_H_init is not None:
            with torch.no_grad():
                self.W_T.copy_(W_T_init)
                self.W_H.copy_(W_H_init)
        else:
            nn.init.xavier_uniform_(self.W_T)
            nn.init.xavier_uniform_(self.W_H)
        if freeze_ro:
            self.W_T.requires_grad = False
            self.W_H.requires_grad = False
        
        # Seasonal diagonal quadratic and cubic: [basis, n_vars]
        self.B_diag = nn.Parameter(torch.zeros(self.n_basis, n_vars))
        self.C_diag = nn.Parameter(torch.zeros(self.n_basis, n_vars))
        if B_diag_init is not None and C_diag_init is not None:
            with torch.no_grad():
                self.B_diag.copy_(B_diag_init)
                self.C_diag.copy_(C_diag_init)
        else:
            nn.init.xavier_uniform_(self.B_diag)
            nn.init.xavier_uniform_(self.C_diag)
        if freeze_diag:
            self.B_diag.requires_grad = False
            self.C_diag.requires_grad = False

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
    """NXRO-Res: Seasonal linear + residual MLP R_θ(X,t).

    dX/dt = L_θ(t) · X + R_θ([X, φ(t)]) where φ(t) are Fourier features.
    
    Variants:
        - Variant 4: Random initialization (L_basis_init=None, freeze_linear=False)
        - Variant 4a (NXRO-Res-WS-FixL): Warm-start linear and freeze, train only MLP
    """

    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64,
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_linear: bool = False):
        """Initialize NXRO-Res model.
        
        Args:
            n_vars: number of variables
            k_max: maximum harmonic order
            hidden: MLP hidden size
            L_basis_init: optional initial values for L_basis
            freeze_linear: if True, freeze L_basis (for variant 4a)
        """
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis)
        if freeze_linear:
            self.L_basis.requires_grad = False
        
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


class NXROResFullXROModel(nn.Module):
    """NXRO-Res-FullXRO (Variant 4b): Frozen full XRO + trainable residual MLP.
    
    dX/dt = L_XRO(t)·X + N_RO,XRO(T,H,t) + N_Diag,XRO(X,t) + R_θ([X, φ(t)])
    
    All XRO components (L, RO, Diag) are frozen (from XRO fit).
    Only the residual MLP R_θ is trainable.
    
    This is the most conservative hybrid approach: uses XRO exactly as-is,
    adds only neural correction.
    """
    
    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64,
                 L_basis_xro: torch.Tensor = None,
                 W_T_xro: torch.Tensor = None,
                 W_H_xro: torch.Tensor = None,
                 B_diag_xro: torch.Tensor = None,
                 C_diag_xro: torch.Tensor = None):
        """Initialize NXRO-Res-FullXRO model (variant 4b).
        
        Args:
            n_vars: number of variables
            k_max: maximum harmonic order
            hidden: MLP hidden size
            L_basis_xro: Linear operator from XRO (REQUIRED, will be frozen)
            W_T_xro, W_H_xro: RO coefficients from XRO (REQUIRED, will be frozen)
            B_diag_xro, C_diag_xro: Diagonal coefficients from XRO (REQUIRED, will be frozen)
        """
        super().__init__()
        assert L_basis_xro is not None, "Variant 4b requires XRO initialization!"
        assert W_T_xro is not None and W_H_xro is not None, "Variant 4b requires RO coefficients!"
        assert B_diag_xro is not None and C_diag_xro is not None, "Variant 4b requires Diag coefficients!"
        assert n_vars >= 2, "Assumes first two vars are T and H"
        
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_ro = W_T_xro.shape[1]  # infer n_ro from W_T shape
        
        # Register XRO components as buffers (frozen, not trainable)
        self.register_buffer('L_basis_xro', L_basis_xro)
        self.register_buffer('W_T_xro', W_T_xro)
        self.register_buffer('W_H_xro', W_H_xro)
        self.register_buffer('B_diag_xro', B_diag_xro)
        self.register_buffer('C_diag_xro', C_diag_xro)
        
        # Residual MLP: inputs X (n_vars) + time emb (n_basis)
        in_dim = n_vars + self.n_basis
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_vars),
        )
    
    def _phi_ro(self, T: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Build RO basis [T^2, T*H, T^3, T^2*H, T*H^2]."""
        return torch.stack([
            T * T,
            T * H,
            T * T * T,
            T * T * H,
            T * H * H,
        ], dim=-1)
    
    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute dX/dt with frozen XRO base + trainable residual."""
        B = x.shape[0]
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        
        # Frozen XRO linear part
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis_xro)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        
        # Frozen XRO RO part (T/H)
        T = x[:, 0]
        H = x[:, 1]
        phi = self._phi_ro(T, H)  # [B, n_ro]
        cT = torch.einsum('bk,kn->bn', emb, self.W_T_xro)
        cH = torch.einsum('bk,kn->bn', emb, self.W_H_xro)
        dxdt[:, 0] = dxdt[:, 0] + torch.sum(cT * phi, dim=-1)
        dxdt[:, 1] = dxdt[:, 1] + torch.sum(cH * phi, dim=-1)
        
        # Frozen XRO diagonal part
        b_t = torch.einsum('bk,kv->bv', emb, self.B_diag_xro)
        c_t = torch.einsum('bk,kv->bv', emb, self.C_diag_xro)
        dxdt = dxdt + b_t * (x * x) + c_t * (x * x * x)
        
        # Trainable residual
        res_in = torch.cat([x, emb], dim=-1)
        dxdt_res = self.residual(res_in)
        
        return dxdt + dxdt_res


class NXROResidualMixModel(NXRORODiagModel):
    """NXRO-ResidualMix: RO+Diag base plus a small residual MLP scaled by α.

    f(x,t) = f_RO+Diag(x,t) + α · R_θ([x, φ(t)]), with α small (fixed or learnable).
    
    Variants:
        - Variant 5d: Random initialization
        - Variant 5d-WS: Warm-start all physics components, train all
        - Variant 5d-Fix*: Various freezing configurations (see README)
    """

    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64,
                 alpha_init: float = 0.1, alpha_learnable: bool = False, alpha_max: float = 0.5,
                 dropout: float = 0.0,
                 L_basis_init: Optional[torch.Tensor] = None,
                 W_T_init: Optional[torch.Tensor] = None,
                 W_H_init: Optional[torch.Tensor] = None,
                 B_diag_init: Optional[torch.Tensor] = None,
                 C_diag_init: Optional[torch.Tensor] = None,
                 freeze_linear: bool = False,
                 freeze_ro: bool = False,
                 freeze_diag: bool = False):
        """Initialize NXRO-ResidualMix model.
        
        Args:
            n_vars: number of variables
            k_max: maximum harmonic order
            hidden: MLP hidden size
            alpha_init: initial value for alpha scaling parameter
            alpha_learnable: whether alpha is trainable
            alpha_max: maximum value for alpha
            dropout: dropout rate in residual MLP
            L_basis_init, W_T_init, W_H_init, B_diag_init, C_diag_init: warm-start values
            freeze_linear, freeze_ro, freeze_diag: freezing flags
        """
        super().__init__(n_vars=n_vars, k_max=k_max, n_ro=5,
                        L_basis_init=L_basis_init, W_T_init=W_T_init, W_H_init=W_H_init,
                        B_diag_init=B_diag_init, C_diag_init=C_diag_init,
                        freeze_linear=freeze_linear, freeze_ro=freeze_ro, freeze_diag=freeze_diag)
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
    
    Variants:
        - 5a (NXRO-Attentive): Random initialization
        - 5a-WS: Warm-start linear, train all
        - 5a-FixL: Warm-start linear and freeze, train only attention

    Args:
        n_vars: number of variables
        k_max: seasonal harmonics
        d: attention hidden size (keep small, e.g., 16–32)
        dropout: dropout on attention weights/output
        mask_mode: 'th_only' to allow attention mainly from T/H (cols 0/1) plus self; 'full' for all-to-all
        L_basis_init: optional initial values for L_basis
        freeze_linear: if True, freeze L_basis
    """

    def __init__(self, n_vars: int, k_max: int = 2, d: int = 32, dropout: float = 0.0, 
                 mask_mode: str = 'th_only',
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_linear: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.d = d
        self.mask_mode = mask_mode
        
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis)
        if freeze_linear:
            self.L_basis.requires_grad = False
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
                 adj_init: torch.Tensor = None, top_k: int = 2, hidden: int = 0,
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_linear: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.use_fixed_graph = use_fixed_graph
        
        # Seasonal linear operator
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis)
        if freeze_linear:
            self.L_basis.requires_grad = False
        
        # Graph parameters
        if use_fixed_graph:
            if adj_init is None:
                adj_init = torch.eye(n_vars)
            self.register_buffer('A_fixed', adj_init)
        else:
            self.A_param = nn.Parameter(torch.zeros(n_vars, n_vars))
            if adj_init is not None:
                with torch.no_grad():
                    self.A_param.copy_(adj_init)
            else:
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


class NXROTransformerModel(nn.Module):
    """NXRO-Transformer: Pure Transformer Encoder Drift (Variables as Tokens).
    
    dX/dt = Transformer(X, t)
    
    Architecture:
    1. Embed scalars x_i -> d_model
    2. Add Positional Encoding (Variable ID) and Time Embedding (Seasonality)
    3. Transformer Encoder Layers
    4. Project d_model -> 1 (scalar drift dx_i/dt)
    
    No explicit linear operator unless requested (pure data-driven).
    """
    
    def __init__(self, n_vars: int, k_max: int = 2, d_model: int = 32, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.d_model = d_model
        
        # Input projection: scalar -> d_model
        self.in_proj = nn.Linear(1, d_model)
        
        # Variable Positional Embedding (Learnable)
        self.var_emb = nn.Parameter(torch.randn(1, n_vars, d_model))
        
        # Time Embedding Projection: n_basis -> d_model
        self.time_proj = nn.Linear(self.n_basis, d_model)
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                dim_feedforward=dim_feedforward, 
                                                dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: d_model -> 1
        self.out_proj = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute dX/dt.
        x: [B, n_vars]
        t_years: [B]
        """
        B, V = x.shape
        
        # 1. Embed variables
        # x: [B, V] -> [B, V, 1] -> [B, V, d_model]
        h = self.in_proj(x.unsqueeze(-1))
        
        # 2. Add Variable Embedding
        h = h + self.var_emb
        
        # 3. Add Time Embedding (Seasonality)
        # emb: [B, n_basis] -> [B, 1, d_model] (broadcast over V)
        time_emb = fourier_time_embedding(t_years, self.k_max)
        time_emb = self.time_proj(time_emb).unsqueeze(1)
        h = h + time_emb
        
        # 4. Transformer Encoder
        # h: [B, V, d_model]
        h = self.transformer_encoder(h)
        
        # 5. Output Projection
        # h: [B, V, d_model] -> [B, V, 1] -> [B, V]
        dxdt = self.out_proj(h).squeeze(-1)
        
        return dxdt
