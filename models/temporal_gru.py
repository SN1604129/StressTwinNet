import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    """
    Models temporal evolution of stress embeddings.

    Input:
        sequence of embeddings [B, T, D]

    Output:
        predicted next stress state
    """

    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x shape: [B, T, D]
        """

        out, _ = self.gru(x)

        # last timestep
        z = out[:, -1, :]

        logits = self.head(z)

        return logits