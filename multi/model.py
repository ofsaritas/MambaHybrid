"""MambaHybrid: shared Mamba encoder with classifier and reconstruction heads."""

import torch
import torch.nn as nn
from src.models.networks import MambaResidualBlock


class MambaEncoder(nn.Module):
    """Feature tokenization + Mamba blocks + mean pooling."""

    def __init__(self, n_features, d_model=128, depth=4,
                 d_state=16, d_conv=4, expansion=2, dropout=0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv,
                               expand=expansion, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        z = self.value_proj(x.unsqueeze(-1)) + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            z = block(z)
        return self.norm(z).mean(dim=1)


class MambaDecoder(nn.Module):
    """Reconstructs input features from the shared encoder embedding."""

    def __init__(self, n_features, d_model=128, depth=2,
                 d_state=16, d_conv=4, expansion=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)
        self.blocks = nn.ModuleList([
            MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv,
                               expand=expansion, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, emb):
        z = emb.unsqueeze(1).expand(-1, self.n_features, -1) + self.pos_embed_dec
        for block in self.blocks:
            z = block(z)
        return self.proj(self.norm(z)).squeeze(-1)


class MambaHybrid(nn.Module):
    """Joint classifier + reconstruction-based anomaly detector."""

    def __init__(self, n_features, n_classes, d_model=128, depth=4,
                 d_state=16, d_conv=4, expansion=2, dropout=0.1,
                 decoder_depth=2, **kw):
        super().__init__()
        self.encoder = MambaEncoder(n_features, d_model, depth,
                                    d_state, d_conv, expansion, dropout)
        self.classifier = nn.Linear(d_model, n_classes)
        self.decoder = MambaDecoder(n_features, d_model, decoder_depth,
                                    d_state, d_conv, expansion, dropout)

    def forward(self, x, return_recon=False):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        if return_recon:
            recon = self.decoder(emb)
            return logits, recon
        return logits

    def anomaly_score(self, x):
        """Per-sample reconstruction MSE."""
        emb = self.encoder(x)
        recon = self.decoder(emb)
        return ((x - recon) ** 2).mean(dim=1)

    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)
