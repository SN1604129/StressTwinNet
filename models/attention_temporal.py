from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    Small 1D encoder for a single modality window.
    Input:  [B, 1, T]
    Output: [B, latent_dim]
    """
    def __init__(self, input_channels: int = 1, latent_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)              # [B, 64, 1]
        x = x.squeeze(-1)            # [B, 64]
        x = self.proj(x)             # [B, latent_dim]
        return x


class ModalityAttentionFusion(nn.Module):
    """
    Learns attention weights over modality embeddings.
    Input:  list of [B, latent_dim]
    Output:
        fused: [B, latent_dim]
        attn:  [B, M]
    """
    def __init__(self, latent_dim: int, num_modalities: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1)
        )
        self.num_modalities = num_modalities

    def forward(self, modality_latents: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # stack -> [B, M, D]
        stacked = torch.stack(modality_latents, dim=1)

        # scores -> [B, M, 1] -> [B, M]
        scores = self.score(stacked).squeeze(-1)
        attn = torch.softmax(scores, dim=1)

        # weighted sum
        fused = torch.sum(stacked * attn.unsqueeze(-1), dim=1)   # [B, D]
        return fused, attn


class TemporalGRUHead(nn.Module):
    """
    Input:  [B, S, D]
    Output: logits [B, num_classes]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        out, _ = self.gru(x)
        h_last = out[:, -1, :]       # [B, H]
        logits = self.classifier(h_last)
        return logits


class AttentionTemporalStressTwinNet(nn.Module):
    """
    Full model:
      per modality encoder -> attention fusion per timestep -> temporal GRU -> classifier

    Expected input:
      x: [B, S, M, T]
         B = batch
         S = sequence length
         M = number of modalities
         T = window length
    """
    def __init__(
        self,
        modality_names: List[str],
        latent_dim: int = 64,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(input_channels=1, latent_dim=latent_dim)
            for name in modality_names
        })

        self.fusion = ModalityAttentionFusion(
            latent_dim=latent_dim,
            num_modalities=self.num_modalities,
        )

        self.temporal = TemporalGRUHead(
            input_dim=latent_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, S, M, T]
        returns:
            logits: [B, num_classes]
            attn_mean: [B, M] average attention across sequence
        """
        b, s, m, t = x.shape
        assert m == self.num_modalities, f"Expected {self.num_modalities} modalities, got {m}"

        fused_sequence = []
        attn_sequence = []

        for step in range(s):
            x_step = x[:, step, :, :]    # [B, M, T]

            modality_latents = []
            for mi, name in enumerate(self.modality_names):
                x_mod = x_step[:, mi, :].unsqueeze(1)   # [B, 1, T]
                z_mod = self.encoders[name](x_mod)      # [B, D]
                modality_latents.append(z_mod)

            fused_t, attn_t = self.fusion(modality_latents)
            fused_sequence.append(fused_t)
            attn_sequence.append(attn_t)

        fused_sequence = torch.stack(fused_sequence, dim=1)   # [B, S, D]
        attn_sequence = torch.stack(attn_sequence, dim=1)     # [B, S, M]

        logits = self.temporal(fused_sequence)
        attn_mean = attn_sequence.mean(dim=1)                 # [B, M]

        return logits, attn_mean