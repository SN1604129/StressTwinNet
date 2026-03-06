import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN feature extractor that maps (B,C,T) -> (B, embed_dim)
    """
    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Optionally compress 128 -> embed_dim (keep embed_dim=128 to match baseline)
        if embed_dim == 128:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(128, embed_dim),
                nn.ReLU(),
            )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x).squeeze(-1)  # (B,128)
        z = self.proj(z)               # (B,embed_dim)
        return z


class EncoderWithHead(nn.Module):
    """
    Encoder + classification head (temporary) so we can train embeddings with labels.
    Returns: logits, z
    """
    def __init__(self, in_channels: int, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logits = self.head(z)
        return logits, z