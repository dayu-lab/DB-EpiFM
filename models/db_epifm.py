import torch
import torch.nn as nn
import torch.nn.functional as F

from .db_epifm_st import DBEpiFMSTBackbone
from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder


class DBEpiFM(nn.Module):
    """Dual-view EEG foundation backbone for pretraining.

    View-T (Spatial-Temporal / ST):
      - Conv-only patch embedding (no FFT)
      - Criss-Cross Transformer over (channels, time-patches)

    View-F (Spatial-Frequency / SF):
      - Band-feature tokens (channels, frequency-bands)
      - Criss-Cross Transformer over (channels, bands)

    Pretraining objectives:
      - L_T: temporal MAE reconstruction over masked time-patches
      - L_F: spectral MAE reconstruction over masked frequency-band tokens
      - L_align: align pooled ST vs SF representations

    Finetune/inference:
      - Return fused latent features (inject pooled SF into ST tokens).
    """

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        seq_len: int = 30,
        n_layer: int = 12,
        nhead: int = 8,
        sf_fs: float = 200.0,
        sf_bands: list[tuple[float, float]] | None = None,
    ):
        super().__init__()

        # ---- ST backbone (Conv-only; no FFT) ----
        self.st = DBEpiFMSTBackbone(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=seq_len,
            n_layer=n_layer,
            nhead=nhead,
        )

        self.d_model = d_model
        self.in_dim = in_dim
        self.sf_fs = float(sf_fs)
        self.sf_bands = sf_bands or [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
        self.n_bands = len(self.sf_bands)
        self.register_buffer(
            "sf_band_masks",
            self._build_band_masks(in_dim, self.sf_fs, self.sf_bands),
            persistent=False,
        )

        # ---- SF Transformer (separate weights; symmetric to ST) ----
        encoder_layer_sf = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.sf_encoder = TransformerEncoder(encoder_layer_sf, num_layers=n_layer, enable_nested_tensor=False)

        # Tokenize band features: each (channel, band) has a small interpretable vector.
        # We use 2 features per band:
        #   - log(absolute band power)
        #   - relative band power (normalized across all bands)
        self.sf_token_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Dropout(0.1),
        )

        # ACPE-like positional encoding over (channel, band)
        self.sf_positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )

        # Reconstruct the 2 band features
        self.sf_decoder = nn.Linear(d_model, 2)

        # ---- Alignment heads ----
        self.align_t = nn.Linear(d_model, d_model)
        self.align_f = nn.Linear(d_model, d_model)

        # ---- Fusion ----
        self.fuse_alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    # --------------------------
    # Utilities
    # --------------------------
    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _build_band_masks(
        patch_size: int,
        fs: float,
        bands: list[tuple[float, float]],
    ) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(patch_size, d=1.0 / fs)
        masks = []
        for f_lo, f_hi in bands:
            lo = max(float(f_lo), 0.0)
            hi = min(float(f_hi), fs / 2.0)
            masks.append((freqs >= lo) & (freqs < hi))
        return torch.stack(masks, dim=0)

    def _band_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute interpretable band features.

        x: (B, C, S, T)
        returns: (B, C, BANDS, 2)

        Feature #1: log(absolute band power)
        Feature #2: relative band power (normalized across all bands)

        We compute power spectrum per patch and then average over the patch axis S
        to obtain a stable band summary for SF tokens.
        """
        b, c, s, t = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Power spectrum: |rfft|^2
        x2 = x.contiguous().view(b * c * s, t)
        spec = torch.fft.rfft(x2, dim=-1, norm='forward')
        psd = (spec.real ** 2 + spec.imag ** 2).view(b, c, s, -1)  # (B,C,S,F)

        if t == self.in_dim and self.sf_band_masks.shape[-1] == psd.shape[-1]:
            band_masks = self.sf_band_masks.to(device=psd.device)
        else:
            band_masks = self._build_band_masks(t, self.sf_fs, self.sf_bands).to(device=psd.device)

        band_powers = []
        for mask in band_masks:
            if mask.any():
                bp = psd[..., mask].mean(dim=-1)  # (B,C,S)
            else:
                bp = torch.zeros((b, c, s), device=psd.device, dtype=psd.dtype)
            band_powers.append(bp)

        # (B,C,S,B)
        bp = torch.stack(band_powers, dim=-1)
        # average over patches S -> (B,C,B)
        bp = bp.mean(dim=2)

        # absolute + relative
        abs_log = torch.log1p(bp)
        rel = bp / (bp.sum(dim=-1, keepdim=True) + 1e-8)

        feats = torch.stack([abs_log, rel], dim=-1)  # (B,C,B,2)
        return feats

    # --------------------------
    # Pretrain forward
    # --------------------------
    def forward_pretrain(
        self,
        x: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_f: torch.Tensor | None = None,
    ):
        """Return recon_t, recon_f, feats_t, feats_f, fused_feats, band_target."""

        # ---- ST branch ----
        st_patch_emb = self.st.patch_embedding(x, mask_t)
        feats_t = self.st.encoder(st_patch_emb)  # (B,C,S,D)
        recon_t = self.st.proj_out(feats_t)      # (B,C,S,T)

        # ---- SF branch ----
        band_target = self._band_features(x)  # (B,C,B,2)
        if mask_f is None:
            band_in = band_target
        else:
            band_in = band_target.clone()
            band_in[mask_f == 1] = 0.0

        sf_tok = self.sf_token_proj(band_in)  # (B,C,B,D)
        pos = self.sf_positional_encoding(sf_tok.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sf_tok = sf_tok + pos
        feats_f = self.sf_encoder(sf_tok)     # (B,C,B,D)
        recon_f = self.sf_decoder(feats_f)    # (B,C,B,2)

        # ---- Fuse ----
        f_pool = feats_f.mean(dim=2)          # (B,C,D)
        fused_feats = feats_t + self.fuse_alpha * f_pool.unsqueeze(2)

        return recon_t, recon_f, feats_t, feats_f, fused_feats, band_target

    # --------------------------
    # Finetune / inference forward
    # --------------------------
    def forward(self, x: torch.Tensor):
        # ST tokens
        st_patch_emb = self.st.patch_embedding(x, mask=None)
        feats_t = self.st.encoder(st_patch_emb)

        # SF pooled
        band = self._band_features(x)
        sf_tok = self.sf_token_proj(band)
        pos = self.sf_positional_encoding(sf_tok.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sf_tok = sf_tok + pos
        feats_f = self.sf_encoder(sf_tok)
        f_pool = feats_f.mean(dim=2)

        return feats_t + self.fuse_alpha * f_pool.unsqueeze(2)

    # --------------------------
    # Loss helpers
    # --------------------------
    def alignment_loss(self, feats_t: torch.Tensor, feats_f: torch.Tensor) -> torch.Tensor:
        # Pool to (B, D)
        z_t = feats_t.mean(dim=(1, 2))
        z_f = feats_f.mean(dim=(1, 2))

        z_t = self._l2_normalize(self.align_t(z_t))
        z_f = self._l2_normalize(self.align_f(z_f))
        return F.mse_loss(z_t, z_f)
