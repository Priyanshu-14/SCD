import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    """
    Placeholder encoder for RGB frames.
    Input: (B, T, C, H, W)
    Output: (B, T, D)
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # Instead of CNN, just return dummy tensor
        B, T, C, H, W = x.shape
        return torch.randn(B, T, self.embed_dim, device=x.device)
