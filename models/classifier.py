import torch
import torch.nn as nn

class SceneChangeClassifier(nn.Module):
    """
    Final classification head for scene change detection.
    Input:  (B, T, D_in) temporal features
    Output: (B, T, num_classes) logits per frame
    """
    def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_in) temporal features
        Returns:
            logits: (B, T, num_classes)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
