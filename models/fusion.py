import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """
    Fusion layer that combines multiple modality embeddings.
    - Accepts variable modalities (some may be missing).
    - Concatenates available features along last dim.
    - Projects into a unified embedding space.
    """
    def __init__(self, input_dims, fusion_dim=256):
        """
        Args:
            input_dims (dict): mapping from modality_name -> embedding dim
                Example: {"visual": 128, "motion": 64, "flow": 64, 
                          "edge": 32, "hist": 32, "coding": 32}
            fusion_dim (int): output embedding dimension after fusion
        """
        super().__init__()
        self.input_dims = input_dims
        total_dim = sum(input_dims.values())
        self.proj = nn.Linear(total_dim, fusion_dim)

    def forward(self, features_dict):
        """
        Args:
            features_dict (dict): 
                Each entry is (B, T, D) tensor or None if missing.
                Example:
                {
                    "visual": (B, T, 128),
                    "motion": (B, T, 64),
                    "flow":   None,   # optional
                    ...
                }
        Returns:
            fused (B, T, fusion_dim)
        """
        B, T = None, None
        feats = []

        for name, feat in features_dict.items():
            if feat is not None:
                if B is None:  # record batch/time from first tensor
                    B, T = feat.shape[:2]
                feats.append(feat)
            else:
                # If missing modality, fill with zeros
                dim = self.input_dims[name]
                feats.append(torch.zeros(B, T, dim, device=list(features_dict.values())[0].device))

        fused = torch.cat(feats, dim=-1)   # (B, T, total_dim)
        fused = self.proj(fused)           # (B, T, fusion_dim)
        return fused
