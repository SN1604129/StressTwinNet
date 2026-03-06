import torch
import torch.nn as nn

from models.multimodal_attention import ModalityAttentionFusion


class SingleModalityEncoder(nn.Module):
    """
    Small 1D CNN for a single modality input [B, 1, T] -> [B, D]
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
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

        if embed_dim == 128:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(128, embed_dim)

        self.embed_dim = embed_dim

    def forward(self, x):
        # x: [B, 1, T]
        z = self.trunk(x).squeeze(-1)   # [B, 128]
        z = self.proj(z)                # [B, D]
        return z


class StressTwinNetAttention(nn.Module):
    """
    Separate modality encoders + attention fusion + classifier.

    Expected channel order:
        [ECG, EDA, RESP, BVP]
    """
    def __init__(self, embed_dim: int = 64, num_classes: int = 2):
        super().__init__()

        self.ecg_encoder = SingleModalityEncoder(embed_dim=embed_dim)
        self.eda_encoder = SingleModalityEncoder(embed_dim=embed_dim)
        self.resp_encoder = SingleModalityEncoder(embed_dim=embed_dim)
        self.bvp_encoder = SingleModalityEncoder(embed_dim=embed_dim)

        self.fusion = ModalityAttentionFusion(embed_dim=embed_dim, num_modalities=4)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: [B, 4, T]
           channel 0 = ECG
           channel 1 = EDA
           channel 2 = RESP
           channel 3 = BVP
        """
        ecg = x[:, 0:1, :]
        eda = x[:, 1:2, :]
        resp = x[:, 2:3, :]
        bvp = x[:, 3:4, :]

        z_ecg = self.ecg_encoder(ecg)
        z_eda = self.eda_encoder(eda)
        z_resp = self.resp_encoder(resp)
        z_bvp = self.bvp_encoder(bvp)

        z_fused, attn_weights = self.fusion([z_ecg, z_eda, z_resp, z_bvp])

        logits = self.classifier(z_fused)

        return logits, z_fused, attn_weights