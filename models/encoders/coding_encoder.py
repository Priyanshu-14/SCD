import torch
import torch.nn as nn

class CodingEncoder(nn.Module):
    """
    Placeholder encoder for video coding features.
    Input: (B, T, num_features)
    Output: (B, T, D)
    """
    def __init__(self, input_dim=10, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, D_in = x.shape
        return torch.randn(B, T, self.embed_dim, device=x.device)
