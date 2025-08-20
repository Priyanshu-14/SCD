import torch
import torch.nn as nn

class EdgeEncoder(nn.Module):
    """
    Placeholder encoder for edge maps.
    Input: (B, T, 1, H, W)
    Output: (B, T, D)
    """
    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C, H, W = x.shape
        return torch.randn(B, T, self.embed_dim, device=x.device)
