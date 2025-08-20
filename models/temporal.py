import torch
import torch.nn as nn
from typing import Optional, Tuple

# ---------------------------
# Helpers
# ---------------------------
def _pack_sequence(x: torch.Tensor, lengths: Optional[torch.Tensor]):
    """
    x: (B, T, D)
    lengths: (B,) actual lengths, or None to treat all as full length T
    """
    if lengths is None:
        return x, None, False
    # sort by length desc for RNN pack
    lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
    x_sorted = x.index_select(0, idx_sort)
    packed = nn.utils.rnn.pack_padded_sequence(
        x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
    )
    return packed, (idx_sort, lengths_sorted), True

def _unpack_sequence(y, pack_ctx, was_packed: bool, T: int):
    """
    y: packed output or (B, T, D)
    pack_ctx: (idx_sort, lengths_sorted) or None
    """
    if not was_packed:
        return y
    y_unpacked, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=T)
    idx_sort, _ = pack_ctx
    # restore original order
    idx_unsort = torch.empty_like(idx_sort)
    idx_unsort[idx_sort] = torch.arange(idx_sort.size(0), device=idx_sort.device)
    return y_unpacked.index_select(0, idx_unsort)

# ---------------------------
# BiLSTM Temporal Modeling
# ---------------------------
class BiLSTMTemporal(nn.Module):
    """
    Temporal module using a bidirectional LSTM.
    Input:  (B, T, D_in)
    Output: (B, T, D_out) with D_out = hidden
    """
    def __init__(self,
                 input_dim: int,
                 hidden: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # project 2*hidden (bi) -> hidden
        self.proj = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D_in)
        lengths: (B,) valid lengths; if None, uses full T for all
        """
        B, T, _ = x.shape
        packed, ctx, was_packed = _pack_sequence(x, lengths)
        y, _ = self.lstm(packed)
        y = _unpack_sequence(y, ctx, was_packed, T)   # (B, T, 2*hidden)
        y = self.proj(y)
        y = self.norm(y)
        return y  # (B, T, hidden)

# ---------------------------
# Lightweight Temporal Transformer (optional)
# ---------------------------
class TemporalTransformer(nn.Module):
    """
    Small Transformer encoder for temporal modeling.
    Input:  (B, T, D_in)
    Output: (B, T, D_out) with D_out = model_dim
    """
    def __init__(self,
                 input_dim: int,
                 model_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D_in)
        lengths: (B,) valid lengths; masks padded positions for attention
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, model_dim)

        # build attention mask: True = masked (ignore)
        if lengths is None:
            key_padding_mask = None
        else:
            device = x.device
            arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            key_padding_mask = arange >= lengths.unsqueeze(1)  # (B, T) bool

        y = self.encoder(x, src_key_padding_mask=key_padding_mask)
        y = self.norm(y)
        return y  # (B, T, model_dim)

# ---------------------------
# Wrapper to switch backends
# ---------------------------
class TemporalBackbone(nn.Module):
    """
    Wrapper that lets you choose 'bilstm' (default) or 'transformer'.
    """
    def __init__(self,
                 input_dim: int,
                 backend: str = "bilstm",
                 hidden: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 transformer_heads: int = 4,
                 transformer_mlp_ratio: float = 4.0):
        super().__init__()
        backend = backend.lower()
        if backend == "bilstm":
            self.mod = BiLSTMTemporal(
                input_dim=input_dim,
                hidden=hidden,
                num_layers=num_layers,
                dropout=dropout
            )
            self.out_dim = hidden
        elif backend == "transformer":
            self.mod = TemporalTransformer(
                input_dim=input_dim,
                model_dim=hidden,
                num_layers=num_layers,
                num_heads=transformer_heads,
                mlp_ratio=transformer_mlp_ratio,
                dropout=dropout
            )
            self.out_dim = hidden
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'bilstm' or 'transformer'.")

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mod(x, lengths=lengths)
