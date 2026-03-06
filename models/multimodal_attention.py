import torch
import torch.nn as nn


class ModalityAttentionFusion(nn.Module):
    """
    Fuses modality embeddings using learned attention weights.

    Input:
        modality_embeddings: list of tensors, each [B, D]

    Output:
        z_fused: [B, D]
        attn_weights: [B, M]
    """

    def __init__(self, embed_dim: int, num_modalities: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, modality_embeddings):
        """
        modality_embeddings: list of [B, D], length = M
        """
        # stack -> [B, M, D]
        Z = torch.stack(modality_embeddings, dim=1)

        # score each modality independently
        scores = []
        for i in range(self.num_modalities):
            zi = Z[:, i, :]                  # [B, D]
            si = self.score_net(zi)          # [B, 1]
            scores.append(si)

        scores = torch.cat(scores, dim=1)    # [B, M]
        attn_weights = torch.softmax(scores, dim=1)  # [B, M]

        # weighted sum
        z_fused = torch.sum(Z * attn_weights.unsqueeze(-1), dim=1)  # [B, D]

        return z_fused, attn_weights