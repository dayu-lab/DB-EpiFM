import torch
import torch.nn as nn
import torch.nn.functional as F

from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder


class DBEpiFMSTBackbone(nn.Module):
    """Spatial-temporal backbone used by DB-EpiFM without patch-level FFT features."""

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        seq_len: int = 30,
        n_layer: int = 12,
        nhead: int = 8,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbeddingConvOnly(in_dim=in_dim, out_dim=out_dim, d_model=d_model, seq_len=seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Linear(d_model, out_dim)
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        return self.proj_out(feats)


class PatchEmbeddingConvOnly(nn.Module):
    """Conv-based patch embedding + ACPE positional encoding (no FFT spectral embedding)."""

    def __init__(self, in_dim: int, out_dim: int, d_model: int, seq_len: int):
        super().__init__()
        self.d_model = d_model

        # ACPE: depth-wise conv positional encoding over (channel, patch)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        # Time-domain convolutional encoder used by the DB-EpiFM temporal branch
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # (B, C, S, T) -> (B, 1, C*S, T)
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return patch_emb + pos


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
