"""Network modules for MambaHybrid and baseline IDS models."""

import warnings
import math
import torch
import torch.nn.functional as F
from torch import nn

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        from mamba_ssm import Mamba as _MambaSSM
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    import selective_scan_cuda  # noqa: F401
    CUDA_KERNEL_AVAILABLE = True
except ImportError:
    CUDA_KERNEL_AVAILABLE = False


@torch.jit.script
def _selective_scan_jit(
    dA: torch.Tensor,
    dB: torch.Tensor,
    x_b: torch.Tensor,
    C_ssm: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """Selective SSM scan (TorchScript)."""
    B, L, d_inner = x_b.shape
    d_state = dA.shape[3]
    h = torch.zeros(B, d_inner, d_state, dtype=x_b.dtype, device=x_b.device)
    y = torch.zeros(B, L, d_inner, dtype=x_b.dtype, device=x_b.device)
    for i in range(L):
        h = dA[:, i] * h + dB[:, i] * x_b[:, i].unsqueeze(-1)
        y[:, i] = (h * C_ssm[:, i].unsqueeze(1)).sum(-1)
    return y


class _SelectiveSSM(nn.Module):
    """Pure-PyTorch selective SSM (Gu & Dao, 2023). Used when CUDA kernel is unavailable."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: str = 'auto', dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_b, z = xz.chunk(2, dim=-1)
        x_b = self.conv1d(x_b.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_b = F.silu(x_b)
        x_dbl = self.x_proj(x_b)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B_ssm.float())
        y = _selective_scan_jit(dA, dB, x_b.float(), C_ssm.float(), self.D)
        y = y.to(x_b.dtype) + x_b * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


def _make_mamba_block(d_model: int, d_state: int, d_conv: int,
                      expand: int, dropout: float) -> nn.Module:
    if MAMBA_AVAILABLE and CUDA_KERNEL_AVAILABLE:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return _MambaSSM(d_model=d_model, d_state=d_state,
                             d_conv=d_conv, expand=expand)
    return _SelectiveSSM(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand, dropout=dropout)


class MambaResidualBlock(nn.Module):
    """Pre-LN residual Mamba block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _make_mamba_block(d_model, d_state, d_conv, expand, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mamba(self.norm(x)))


class BiMambaResidualBlock(nn.Module):
    """Bidirectional Pre-LN residual Mamba block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = _make_mamba_block(d_model, d_state, d_conv, expand, dropout)
        self.mamba_bwd = _make_mamba_block(d_model, d_state, d_conv, expand, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        fwd = self.mamba_fwd(h)
        bwd = self.mamba_bwd(h.flip(1)).flip(1)
        return x + self.drop(fwd + bwd)


class BiMambaIDS(nn.Module):
    """Bidirectional Mamba classifier for tabular flow features."""

    def __init__(self, n_features: int, n_classes: int,
                 d_model: int = 128, depth: int = 4,
                 d_state: int = 16, d_conv: int = 4,
                 expansion: int = 2, dropout: float = 0.1, **kw):
        super().__init__()
        self.n_features = n_features
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            BiMambaResidualBlock(d_model, d_state, d_conv, expansion, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            z = block(z)
        emb = self.norm(z).mean(dim=1)
        logits = self.head(emb)
        return (logits, emb) if return_embedding else logits


class MambaIDS(nn.Module):
    """Unidirectional Mamba classifier for tabular flow features."""

    def __init__(self, n_features: int, n_classes: int,
                 d_model: int = 128, depth: int = 4,
                 d_state: int = 16, d_conv: int = 4,
                 expansion: int = 2, dropout: float = 0.1, **kw):
        super().__init__()
        self.n_features = n_features
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, d_state, d_conv, expansion, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            z = block(z)
        emb = self.norm(z).mean(dim=1)
        logits = self.head(emb)
        return (logits, emb) if return_embedding else logits


class _GatedConvBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 2):
        super().__init__()
        hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        a, b = self.in_proj(x).chunk(2, dim=-1)
        a = self.dwconv(a.transpose(1, 2)).transpose(1, 2)
        a = torch.tanh(a) * torch.sigmoid(self.gate(b))
        return r + self.drop(self.out(a))


class GatedConvIDS(nn.Module):
    """Gated depthwise-conv baseline."""

    def __init__(self, n_features: int, n_classes: int,
                 d_model: int = 128, depth: int = 4,
                 expansion: int = 2, dropout: float = 0.1, **kw):
        super().__init__()
        self.n_features = n_features
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.Sequential(*[
            _GatedConvBlock(d_model, dropout, expansion) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        z = self.blocks(z)
        emb = self.norm(z).mean(dim=1)
        logits = self.head(emb)
        return (logits, emb) if return_embedding else logits


class IDSMLP(nn.Module):
    """MLP baseline."""

    def __init__(self, n_features: int, n_classes: int,
                 d_model: int = 128, depth: int = 3,
                 dropout: float = 0.1, **kw):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(depth):
            layers += [
                nn.Linear(in_dim, d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = d_model
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.encoder(x)
        logits = self.head(z)
        return (logits, z) if return_embedding else logits


class LSTMIDS(nn.Module):
    """Bidirectional LSTM baseline."""

    def __init__(self, n_features: int, n_classes: int,
                 d_model: int = 128, depth: int = 4,
                 dropout: float = 0.1, **kw):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=depth,
            batch_first=True, bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        z, _ = self.lstm(z)
        emb = self.norm(z).mean(dim=1)
        emb = self.drop(emb)
        logits = self.head(emb)
        return (logits, emb) if return_embedding else logits


class TransformerIDS(nn.Module):
    """Transformer encoder baseline."""

    def __init__(self, n_features: int, n_classes: int,
                 d_model: int = 128, depth: int = 4,
                 dropout: float = 0.1, **kw):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        z = self.encoder(z)
        emb = self.norm(z).mean(dim=1)
        emb = self.drop(emb)
        logits = self.head(emb)
        return (logits, emb) if return_embedding else logits


def build_model(name: str, n_features: int, n_classes: int, cfg: dict) -> nn.Module:
    mc = cfg.get('model', {})
    d_model = mc.get('d_model', 128)
    depth = mc.get('depth', 4)
    dropout = mc.get('dropout', 0.1)
    expansion = mc.get('expansion', 2)
    d_state = mc.get('d_state', 16)
    d_conv = mc.get('d_conv', 4)

    if name == 'lstm_ids':
        return LSTMIDS(n_features, n_classes, d_model=d_model,
                       depth=depth, dropout=dropout)
    if name == 'transformer_ids':
        return TransformerIDS(n_features, n_classes, d_model=d_model,
                              depth=depth, dropout=dropout)
    if name in ('ids_mlp', 'ids_mlp_no_focal'):
        return IDSMLP(n_features, n_classes, d_model=d_model,
                      depth=3, dropout=dropout)
    if name in ('gated_conv_ids', 'mamba_no_ssm'):
        return GatedConvIDS(n_features, n_classes, d_model=d_model,
                            depth=depth, expansion=expansion, dropout=dropout)
    if name == 'mamba_small':
        return MambaIDS(
            n_features, n_classes,
            d_model=mc.get('small_d_model', 64),
            depth=mc.get('small_depth', 2),
            d_state=d_state, d_conv=d_conv,
            expansion=expansion, dropout=dropout,
        )
    if name == 'bimamba_ids':
        return BiMambaIDS(
            n_features, n_classes, d_model=d_model, depth=depth,
            d_state=d_state, d_conv=d_conv,
            expansion=expansion, dropout=dropout,
        )
    return MambaIDS(
        n_features, n_classes, d_model=d_model, depth=depth,
        d_state=d_state, d_conv=d_conv,
        expansion=expansion, dropout=dropout,
    )
