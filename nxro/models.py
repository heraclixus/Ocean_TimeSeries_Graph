import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class PureNeuralODEModel(nn.Module):
    """Pure Neural ODE: Learns dynamics entirely from data with no physical priors.

    dX/dt = G_θ(X)  or  dX/dt = G_θ([X, t])

    This is a baseline model with NO XRO structure:
    - No seasonal linear operator L(t)
    - No Fourier time embeddings (unless use_time=True)
    - No physics-based nonlinear terms
    - Just a pure MLP mapping state to drift

    Args:
        n_vars: number of variables
        hidden: MLP hidden size
        depth: number of hidden layers
        dropout: dropout rate inside MLP
        use_time: if True, include normalized time as input feature (but NOT seasonal Fourier features)
    """

    def __init__(self, n_vars: int, hidden: int = 64, depth: int = 2,
                 dropout: float = 0.0, use_time: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.use_time = use_time

        # Input dimension: state variables + optionally time
        in_dim = n_vars + (1 if use_time else 0)
        
        # Build MLP
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            layers += [nn.Tanh()]
        layers += [nn.Linear(hidden, n_vars)]
        self.drift = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute f(x,t) = dX/dt.

        Args:
            x: [B, n_vars] state
            t_years: [B] time in years (used only if use_time=True, as normalized scalar)
        Returns:
            dxdt: [B, n_vars]
        """
        if self.use_time:
            # Normalize time to roughly [-1, 1] range (centered around 2000)
            t_norm = ((t_years - 2000.0) / 20.0).unsqueeze(-1)  # [B, 1]
            drift_in = torch.cat([x, t_norm], dim=-1)
        else:
            drift_in = x
        return self.drift(drift_in)


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
                 freeze_linear: bool = False,
                 disable_seasonal_gate: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.d = d
        self.mask_mode = mask_mode
        self.disable_seasonal_gate = disable_seasonal_gate
        
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
        if self.disable_seasonal_gate:
            dxdt = dxdt + out
        else:
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


class _NXROMemoryBackbone(nn.Module):
    """Shared lagged-linear backbone for memory-aware NXRO models."""

    def __init__(self, n_vars: int, memory_depth: int, k_max: int = 2,
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_instantaneous: bool = False,
                 freeze_lagged: bool = False,
                 init_lagged_zero: bool = True):
        super().__init__()
        if memory_depth < 0:
            raise ValueError(f"memory_depth must be >= 0, got {memory_depth}")

        self.n_vars = n_vars
        self.memory_depth = memory_depth
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.n_lags = memory_depth + 1

        self.L_basis_0 = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            if L_basis_init.shape != (self.n_basis, n_vars, n_vars):
                raise ValueError(
                    f"L_basis_init shape {L_basis_init.shape} must match "
                    f"({self.n_basis}, {n_vars}, {n_vars})"
                )
            with torch.no_grad():
                self.L_basis_0.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis_0)

        if self.memory_depth > 0:
            self.L_basis_memory = nn.Parameter(
                torch.zeros(self.memory_depth, self.n_basis, n_vars, n_vars)
            )
            if init_lagged_zero:
                with torch.no_grad():
                    self.L_basis_memory.zero_()
            else:
                nn.init.xavier_uniform_(self.L_basis_memory)
        else:
            self.register_parameter('L_basis_memory', None)

        if freeze_instantaneous:
            self.L_basis_0.requires_grad = False
        if freeze_lagged and self.L_basis_memory is not None:
            self.L_basis_memory.requires_grad = False

    def _unpack_history(self, x_history: torch.Tensor, t_history: torch.Tensor):
        if x_history.dim() != 3:
            raise ValueError(
                f"x_history must have shape [B, P+1, V], got {tuple(x_history.shape)}"
            )
        if t_history.dim() != 2:
            raise ValueError(
                f"t_history must have shape [B, P+1], got {tuple(t_history.shape)}"
            )
        if x_history.shape[1] != self.n_lags:
            raise ValueError(
                f"Expected history length {self.n_lags}, got {x_history.shape[1]}"
            )
        if x_history.shape[2] != self.n_vars:
            raise ValueError(
                f"Expected {self.n_vars} variables, got {x_history.shape[2]}"
            )
        if t_history.shape[1] != self.n_lags:
            raise ValueError(
                f"Expected time history length {self.n_lags}, got {t_history.shape[1]}"
            )

        # Convert from [oldest ... current] to [current, lag1, ...].
        x_lags = torch.flip(x_history, dims=[1])
        t_current = t_history[:, -1]
        emb = fourier_time_embedding(t_current, self.k_max)
        return x_lags, emb

    def _linear_memory_term(self, x_history: torch.Tensor, t_history: torch.Tensor):
        x_lags, emb = self._unpack_history(x_history, t_history)
        L0_t = torch.einsum('bk,kuv->buv', emb, self.L_basis_0)
        dxdt = torch.einsum('buv,bv->bu', L0_t, x_lags[:, 0, :])

        if self.L_basis_memory is not None:
            Lm_t = torch.einsum('bk,lkuv->bluv', emb, self.L_basis_memory)
            dxdt = dxdt + torch.einsum('bluv,blv->bu', Lm_t, x_lags[:, 1:, :])
        return dxdt, emb, x_lags


class NXROMemoryLinearModel(_NXROMemoryBackbone):
    """Lagged seasonal linear model for memory-aware NXRO."""

    def __init__(self, n_vars: int, memory_depth: int, k_max: int = 2,
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_instantaneous: bool = False,
                 freeze_lagged: bool = False,
                 init_lagged_zero: bool = True):
        super().__init__(
            n_vars=n_vars,
            memory_depth=memory_depth,
            k_max=k_max,
            L_basis_init=L_basis_init,
            freeze_instantaneous=freeze_instantaneous,
            freeze_lagged=freeze_lagged,
            init_lagged_zero=init_lagged_zero,
        )

    def forward(self, x_history: torch.Tensor, t_history: torch.Tensor) -> torch.Tensor:
        dxdt, _, _ = self._linear_memory_term(x_history, t_history)
        return dxdt


class NXROMemoryResModel(_NXROMemoryBackbone):
    """Lagged seasonal linear operator plus residual MLP over history."""

    def __init__(self, n_vars: int, memory_depth: int, k_max: int = 2, hidden: int = 64,
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_instantaneous: bool = False,
                 freeze_lagged: bool = False,
                 init_lagged_zero: bool = True):
        super().__init__(
            n_vars=n_vars,
            memory_depth=memory_depth,
            k_max=k_max,
            L_basis_init=L_basis_init,
            freeze_instantaneous=freeze_instantaneous,
            freeze_lagged=freeze_lagged,
            init_lagged_zero=init_lagged_zero,
        )
        in_dim = self.n_lags * n_vars + self.n_basis
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_vars),
        )

    def forward(self, x_history: torch.Tensor, t_history: torch.Tensor) -> torch.Tensor:
        dxdt_lin, emb, _ = self._linear_memory_term(x_history, t_history)
        res_in = torch.cat([x_history.reshape(x_history.shape[0], -1), emb], dim=-1)
        dxdt_res = self.residual(res_in)
        return dxdt_lin + dxdt_res


class NXROMemoryAttentionModel(_NXROMemoryBackbone):
    """Spatiotemporal attention over lagged variable tokens."""

    def __init__(self, n_vars: int, memory_depth: int, k_max: int = 2, d: int = 32,
                 n_heads: int = 1, dropout: float = 0.0, mask_mode: str = 'th_only',
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_instantaneous: bool = False,
                 freeze_lagged: bool = False,
                 init_lagged_zero: bool = True):
        super().__init__(
            n_vars=n_vars,
            memory_depth=memory_depth,
            k_max=k_max,
            L_basis_init=L_basis_init,
            freeze_instantaneous=freeze_instantaneous,
            freeze_lagged=freeze_lagged,
            init_lagged_zero=init_lagged_zero,
        )
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")

        self.d = d
        self.n_heads = n_heads
        self.mask_mode = mask_mode
        self.token_in = nn.Linear(1, d, bias=False)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d, 1, bias=False)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.var_embedding = nn.Parameter(torch.randn(n_vars, d) * 0.02)
        self.lag_embedding = nn.Parameter(torch.randn(self.n_lags, d) * 0.02)
        self.alpha_w = nn.Parameter(torch.zeros(self.n_basis))
        nn.init.normal_(self.alpha_w, std=0.1)

        allowed = torch.eye(n_vars)
        if mask_mode == 'th_only':
            allowed[:, 0] = 1.0
            if n_vars > 1:
                allowed[:, 1] = 1.0
        else:
            allowed = torch.ones(n_vars, n_vars)
        st_allowed = allowed.repeat(1, self.n_lags)
        self.register_buffer('attn_mask_st', st_allowed < 0.5)

    def forward(self, x_history: torch.Tensor, t_history: torch.Tensor) -> torch.Tensor:
        dxdt_lin, emb, x_lags = self._linear_memory_term(x_history, t_history)
        B = x_lags.shape[0]
        n_tokens = self.n_lags * self.n_vars

        tok = x_lags.reshape(B, n_tokens, 1)
        tok = self.token_in(tok)
        var_pos = self.var_embedding.repeat(self.n_lags, 1)
        lag_pos = self.lag_embedding.repeat_interleave(self.n_vars, dim=0)
        tok = tok + var_pos.unsqueeze(0) + lag_pos.unsqueeze(0)

        query = tok[:, :self.n_vars, :]
        attn_out, _ = self.attn(query, tok, tok, attn_mask=self.attn_mask_st)
        out = self.out_proj(attn_out).squeeze(-1)
        out = self.out_drop(out)
        alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, self.alpha_w)).unsqueeze(-1)
        return dxdt_lin + alpha * out


class NXROMemoryGraphModel(_NXROMemoryBackbone):
    """Lagged seasonal linear operator plus spatiotemporal graph correction."""

    def __init__(self, n_vars: int, memory_depth: int, k_max: int = 2,
                 use_fixed_graph: bool = True, adj_init: torch.Tensor = None,
                 graph_mode: str = 'agg_spatial',
                 L_basis_init: Optional[torch.Tensor] = None,
                 freeze_instantaneous: bool = False,
                 freeze_lagged: bool = False,
                 init_lagged_zero: bool = True):
        super().__init__(
            n_vars=n_vars,
            memory_depth=memory_depth,
            k_max=k_max,
            L_basis_init=L_basis_init,
            freeze_instantaneous=freeze_instantaneous,
            freeze_lagged=freeze_lagged,
            init_lagged_zero=init_lagged_zero,
        )
        if graph_mode not in {'agg_spatial', 'full_st'}:
            raise ValueError(f"Unsupported graph_mode: {graph_mode}")

        self.use_fixed_graph = use_fixed_graph
        self.graph_mode = graph_mode
        if adj_init is None:
            adj_init = torch.eye(n_vars)
        if use_fixed_graph:
            self.register_buffer('A_fixed', adj_init)
        else:
            self.A_param = nn.Parameter(torch.zeros(n_vars, n_vars))
            with torch.no_grad():
                self.A_param.copy_(adj_init)

        self.alpha_w = nn.Parameter(torch.zeros(self.n_basis))
        nn.init.normal_(self.alpha_w, std=0.1)

        if graph_mode == 'agg_spatial':
            self.W_g = nn.Parameter(torch.zeros(n_vars, n_vars))
            nn.init.xavier_uniform_(self.W_g)
            self.lag_gate = nn.Parameter(torch.zeros(self.n_basis, self.n_lags))
            with torch.no_grad():
                self.lag_gate.zero_()
                self.lag_gate[0, 0] = 1.0
        else:
            n_tokens = self.n_lags * n_vars
            self.W_full = nn.Parameter(torch.zeros(n_tokens, n_tokens))
            nn.init.xavier_uniform_(self.W_full)
            if self.memory_depth > 0:
                self.temporal_edge_strength = nn.Parameter(torch.ones(self.memory_depth))
            else:
                self.register_parameter('temporal_edge_strength', None)

    def _normalize_rows(self, A: torch.Tensor) -> torch.Tensor:
        A = A + torch.eye(A.shape[0], device=A.device)
        rowsum = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return A / rowsum

    def _get_spatial_adj(self) -> torch.Tensor:
        if self.use_fixed_graph:
            return self.A_fixed
        return torch.relu(self.A_param)

    def _build_spatiotemporal_adj(self, A_spatial: torch.Tensor) -> torch.Tensor:
        n_tokens = self.n_lags * self.n_vars
        A_st = torch.zeros(n_tokens, n_tokens, device=A_spatial.device, dtype=A_spatial.dtype)
        for lag_idx in range(self.n_lags):
            sl = slice(lag_idx * self.n_vars, (lag_idx + 1) * self.n_vars)
            A_st[sl, sl] = A_spatial
        if self.temporal_edge_strength is not None:
            eye = torch.eye(self.n_vars, device=A_spatial.device, dtype=A_spatial.dtype)
            for lag_idx in range(self.memory_depth):
                weight = torch.relu(self.temporal_edge_strength[lag_idx])
                sl0 = slice(lag_idx * self.n_vars, (lag_idx + 1) * self.n_vars)
                sl1 = slice((lag_idx + 1) * self.n_vars, (lag_idx + 2) * self.n_vars)
                A_st[sl0, sl1] = weight * eye
                A_st[sl1, sl0] = weight * eye
        return A_st

    def forward(self, x_history: torch.Tensor, t_history: torch.Tensor) -> torch.Tensor:
        dxdt_lin, emb, x_lags = self._linear_memory_term(x_history, t_history)
        A_spatial = self._normalize_rows(self._get_spatial_adj())

        if self.graph_mode == 'agg_spatial':
            lag_logits = torch.einsum('bk,kl->bl', emb, self.lag_gate)
            lag_weights = torch.softmax(lag_logits, dim=-1).unsqueeze(-1)
            x_agg = torch.sum(lag_weights * x_lags, dim=1)
            message = torch.matmul(x_agg, A_spatial.T)
            dx_graph = torch.tanh(torch.matmul(message, self.W_g.T))
        else:
            A_st = self._normalize_rows(self._build_spatiotemporal_adj(A_spatial))
            tok = x_lags.reshape(x_lags.shape[0], -1)
            message = torch.matmul(tok, A_st.T)
            dx_tok = torch.tanh(torch.matmul(message, self.W_full.T))
            dx_graph = dx_tok[:, :self.n_vars]

        alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, self.alpha_w)).unsqueeze(-1)
        return dxdt_lin + alpha * dx_graph


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
                 hidden: int = 16, dropout: float = 0.0, use_gat: bool = False,
                 disable_seasonal_gate: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.disable_seasonal_gate = disable_seasonal_gate
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
        if self.disable_seasonal_gate:
            return dxdt + graph_out
        else:
            alpha = torch.sigmoid(torch.einsum('bk,k->b', emb, self.alpha_w)).unsqueeze(-1)
            return dxdt + alpha * graph_out


class NXROTransformerModel(nn.Module):
    """NXRO-Transformer: Pure Transformer model for time series forecasting.
    
    Uses a Transformer encoder to process the state variables and time features,
    then outputs the derivative dX/dt.
    
    Architecture:
        - Input: Concatenate state X with seasonal Fourier features φ(t)
        - Transformer encoder with multi-head self-attention
        - Output projection to predict dX/dt
    
    Args:
        n_vars: number of state variables
        k_max: maximum harmonic order for seasonal features (default 2)
        d_model: dimension of transformer embeddings (default 64)
        nhead: number of attention heads (default 4)
        num_layers: number of transformer encoder layers (default 2)
        dim_feedforward: dimension of feedforward network (default 256)
        dropout: dropout rate (default 0.1)
    """
    
    def __init__(self, n_vars: int, k_max: int = 2, d_model: int = 64, 
                 nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.d_model = d_model
        self.n_basis = 1 + 2 * k_max  # Seasonal features dimension
        
        # Input projection: map [state_value + time_features] to d_model for each variable
        # Each variable gets: its own value (1) + seasonal features (n_basis)
        input_dim = 1 + self.n_basis
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding for variables (treat each variable as a token)
        self.var_pos_embedding = nn.Parameter(torch.randn(n_vars, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: map from d_model back to derivative for each variable
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute f(x,t) = dX/dt using Transformer.
        
        Args:
            x: [B, n_vars] state variables
            t_years: [B] absolute time in years (e.g., 1979 + m/12)
        
        Returns:
            dxdt: [B, n_vars] time derivatives
        """
        B = x.shape[0]
        
        # Get seasonal Fourier features
        time_emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        
        # Expand time features to match each variable
        time_emb_expanded = time_emb.unsqueeze(1).expand(B, self.n_vars, self.n_basis)  # [B, n_vars, n_basis]
        
        # Expand state to add feature dimension
        x_expanded = x.unsqueeze(-1)  # [B, n_vars, 1]
        
        # Concatenate state with time features for each variable
        x_with_time = torch.cat([x_expanded, time_emb_expanded], dim=-1)  # [B, n_vars, 1 + n_basis]
        
        # Reshape for linear projection: [B, n_vars, input_dim] -> [B*n_vars, input_dim]
        x_with_time_flat = x_with_time.reshape(B * self.n_vars, -1)
        
        # Project to d_model
        x_proj_flat = self.input_proj(x_with_time_flat)  # [B*n_vars, d_model]
        
        # Reshape back: [B*n_vars, d_model] -> [B, n_vars, d_model]
        x_proj = x_proj_flat.reshape(B, self.n_vars, self.d_model)
        
        # Add positional encoding (variable-specific)
        x_proj = x_proj + self.var_pos_embedding.unsqueeze(0)  # [B, n_vars, d_model]
        
        # Pass through transformer encoder
        # Treat each variable as a token in the sequence
        transformer_out = self.transformer_encoder(x_proj)  # [B, n_vars, d_model]
        
        # Project to output (derivative for each variable)
        # Reshape for output projection: [B, n_vars, d_model] -> [B*n_vars, d_model]
        transformer_out_flat = transformer_out.reshape(B * self.n_vars, self.d_model)
        dxdt_flat = self.output_proj(transformer_out_flat).squeeze(-1)  # [B*n_vars]
        
        # Reshape back: [B*n_vars] -> [B, n_vars]
        dxdt = dxdt_flat.reshape(B, self.n_vars)
        
        return dxdt


class PureTransformerModel(nn.Module):
    """Pure Transformer: Learns dynamics entirely from data with NO physical priors.
    
    This is a black-box baseline Transformer:
    - NO seasonal Fourier time features
    - NO physical structure
    - Just raw state variables as input tokens
    
    Architecture:
        - Each variable is treated as a token
        - Input: just the state value (optionally + normalized time)
        - Transformer encoder with self-attention
        - Output projection to predict dX/dt
    
    Args:
        n_vars: number of state variables
        d_model: dimension of transformer embeddings (default 64)
        nhead: number of attention heads (default 4)
        num_layers: number of transformer encoder layers (default 2)
        dim_feedforward: dimension of feedforward network (default 256)
        dropout: dropout rate (default 0.1)
        use_time: if True, include normalized time as additional input feature
    """
    
    def __init__(self, n_vars: int, d_model: int = 64, 
                 nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 use_time: bool = False):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.use_time = use_time
        
        # Input projection: map [state_value (+ optional time)] to d_model
        input_dim = 1 + (1 if use_time else 0)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding for variables (treat each variable as a token)
        self.var_pos_embedding = nn.Parameter(torch.randn(n_vars, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: map from d_model back to derivative for each variable
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute f(x,t) = dX/dt using Transformer (NO seasonal features).
        
        Args:
            x: [B, n_vars] state variables
            t_years: [B] absolute time in years (only used if use_time=True)
        
        Returns:
            dxdt: [B, n_vars] time derivatives
        """
        B = x.shape[0]
        
        # Expand state to add feature dimension
        x_expanded = x.unsqueeze(-1)  # [B, n_vars, 1]
        
        if self.use_time:
            # Normalize time to roughly [-1, 1] range (centered around 2000)
            t_norm = ((t_years - 2000.0) / 20.0).unsqueeze(-1).unsqueeze(1)  # [B, 1, 1]
            t_norm = t_norm.expand(B, self.n_vars, 1)  # [B, n_vars, 1]
            x_input = torch.cat([x_expanded, t_norm], dim=-1)  # [B, n_vars, 2]
        else:
            x_input = x_expanded  # [B, n_vars, 1]
        
        # Reshape for linear projection: [B, n_vars, input_dim] -> [B*n_vars, input_dim]
        x_input_flat = x_input.reshape(B * self.n_vars, -1)
        
        # Project to d_model
        x_proj_flat = self.input_proj(x_input_flat)  # [B*n_vars, d_model]
        
        # Reshape back: [B*n_vars, d_model] -> [B, n_vars, d_model]
        x_proj = x_proj_flat.reshape(B, self.n_vars, self.d_model)
        
        # Add positional encoding (variable-specific)
        x_proj = x_proj + self.var_pos_embedding.unsqueeze(0)  # [B, n_vars, d_model]
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x_proj)  # [B, n_vars, d_model]
        
        # Project to output (derivative for each variable)
        transformer_out_flat = transformer_out.reshape(B * self.n_vars, self.d_model)
        dxdt_flat = self.output_proj(transformer_out_flat).squeeze(-1)  # [B*n_vars]
        
        # Reshape back: [B*n_vars] -> [B, n_vars]
        dxdt = dxdt_flat.reshape(B, self.n_vars)
        
        return dxdt


class NXRODeepLearnableGCN(nn.Module):
    """
    Deep Learnable GCN model - best configuration from hyperparameter search.
    
    Key innovations:
    - 3+ layers of GCN (deeper than standard 2-layer)
    - Layer normalization for stable deep training
    - Learnable adjacency matrix with no/low L1 regularization
    - Cosine annealing LR scheduler (handled in training)
    
    This model achieved val_rmse=0.4633, beating previous best of 0.4974.
    """
    
    def __init__(self, n_vars: int, k_max: int = 2, hidden: int = 64, n_layers: int = 3,
                 dropout: float = 0.0, use_layer_norm: bool = True,
                 L_basis_init: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_vars = n_vars
        self.k_max = k_max
        self.n_basis = 1 + 2 * k_max
        self.hidden = hidden
        self.n_layers = n_layers
        self.use_layer_norm = use_layer_norm
        
        # Seasonal linear operator L_basis
        self.L_basis = nn.Parameter(torch.zeros(self.n_basis, n_vars, n_vars))
        if L_basis_init is not None:
            with torch.no_grad():
                self.L_basis.copy_(L_basis_init)
        else:
            nn.init.xavier_uniform_(self.L_basis)
        
        # Learnable adjacency matrix - small random init
        self.A_param = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)
        
        # Seasonal gate for neural contribution
        self.alpha_w = nn.Parameter(torch.zeros(self.n_basis))
        nn.init.normal_(self.alpha_w, std=0.1)
        
        # Build GCN layers
        self.gcn_weights = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        in_dim = 1
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 1
            self.gcn_weights.append(nn.Linear(in_dim, out_dim, bias=False))
            if use_layer_norm and i < n_layers - 1:
                self.layer_norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
    
    def get_adjacency(self) -> torch.Tensor:
        """Get row-normalized adjacency matrix."""
        A = F.relu(self.A_param)
        rowsum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return A / rowsum
    
    def forward(self, x: torch.Tensor, t_years: torch.Tensor) -> torch.Tensor:
        """Compute f(x,t) = dX/dt.
        
        Args:
            x: [B, n_vars] state variables
            t_years: [B] absolute time in years
        
        Returns:
            dxdt: [B, n_vars] time derivatives
        """
        B = x.shape[0]
        t_years = t_years.float()
        emb = fourier_time_embedding(t_years, self.k_max)  # [B, n_basis]
        
        # Linear part: L(t) @ x
        L_t = torch.einsum('bk,kuv->buv', emb, self.L_basis)
        dxdt = torch.einsum('buv,bv->bu', L_t, x)
        
        # GCN part
        A = self.get_adjacency()  # [n_vars, n_vars]
        h = x.unsqueeze(-1)  # [B, n_vars, 1]
        
        for i in range(self.n_layers):
            # Graph convolution: h = A @ h @ W
            h = self.gcn_weights[i](h)  # Linear transform
            h = torch.einsum('ij,bjd->bid', A, h)  # Graph conv
            
            if i < self.n_layers - 1:
                h = F.relu(h)
                if self.use_layer_norm and self.layer_norms:
                    h = self.layer_norms[i](h)
                h = self.dropout(h)
        
        gnn_out = h.squeeze(-1)  # [B, n_vars]
        
        # Seasonal gate α(t) in [0, 1]
        alpha = torch.sigmoid(emb @ self.alpha_w)  # [B]
        
        # Combine: dxdt = L(t)x + α(t) * GNN(x)
        dxdt = dxdt + alpha.unsqueeze(-1) * gnn_out
        
        return dxdt

