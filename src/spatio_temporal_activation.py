import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import time
from typing import Tuple, List
import random
import argparse
import os
import sys
import csv
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Make behaviour more deterministic for reproducibility (best-effort)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# set other seeds
random.seed(42)
torch.cuda.manual_seed_all(42)

# =============================================================================
# 1. SpatioTemporalFourierAdaPolyGLU Implementation (Corrected)
# =============================================================================
# This is a refined version of your module for cleaner integration.
# It now uses `nn.Conv2d` for all projections and a more structured approach.

class SpatioTemporalFourierAdaPolyGLU(nn.Module):
    """SpatioTemporal Fourier + Adaptive Polynomial GLU.

    Clean, well-indented implementation that predicts spatially-varying
    coefficients for polynomial and Fourier basis kernels and applies
    them per-channel. The coefficient predictor consumes a nonlinear
    projection so it receives rich features.
    """
    def __init__(self, d_model: int, d_ff: int,
                 spatial_kernel_size: int = 3,
                 max_1d_degree: int = 2, max_1d_freq: int = 1,
                 use_poly: bool = True, use_fourier: bool = True, use_gate: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.kernel_size = spatial_kernel_size
        self.max_1d_degree = max_1d_degree
        self.max_1d_freq = max_1d_freq

        # Ablation flags
        self.use_poly = use_poly
        self.use_fourier = use_fourier
        self.use_gate = use_gate

        # Basis counts
        self.num_poly_basis = (max_1d_degree + 1) ** 2
        self.num_fourier_basis = 4 * max_1d_freq if max_1d_freq > 0 else 0
        self.total_coeffs = self.num_poly_basis + self.num_fourier_basis

        # Projections
        self.gate_proj = nn.Conv2d(d_model, d_ff, kernel_size=1, bias=True)
        self.up_proj = nn.Conv2d(d_model, d_ff, kernel_size=1, bias=True)
        self.nonlinear_proj = nn.Conv2d(d_model, d_ff, kernel_size=1, bias=True)
        self.down_proj = nn.Conv2d(d_ff, d_model, kernel_size=1, bias=True)

        # Coefficient predictor consumes nonlinear-projected features
        # produces (d_ff * total_coeffs) channels so we can reshape to
        # (B, d_ff, total_coeffs, H, W)
        self.coeff_predictor = nn.Sequential(
            nn.Conv2d(d_ff, d_ff, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2),
            nn.GELU(),
            nn.Conv2d(d_ff, d_ff, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(d_ff, d_ff * self.total_coeffs, kernel_size=1, bias=True)
        )

        # Learnable mixing / gating parameters (small non-zero inits)
        # gamma starts small so adaptive path receives gradients but doesn't explode
        self.gamma_poly = nn.Parameter(torch.tensor(0.01))
        self.gamma_fourier = nn.Parameter(torch.tensor(0.01))
        self.basis_mix_logit = nn.Parameter(torch.tensor(0.0))
        # Do NOT bias strongly toward the linear path initially — allow the
        # learned adaptive (nonlinear) path to contribute early in training.
        self.linear_mix_logit = nn.Parameter(torch.tensor(0.0))

        # Slightly reduced dropout so the adaptive path can learn more robustly
        self.dropout = nn.Dropout2d(0.10)

        self._init_weights()

    def _init_weights(self):
        # Initialize conv layers reasonably
        for layer in [self.gate_proj, self.up_proj, self.nonlinear_proj, self.down_proj]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Coeff predictor convs
        for layer in self.coeff_predictor:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Final coeff predictor: small normal std to avoid large initial coeffs
        nn.init.normal_(self.coeff_predictor[-1].weight, mean=0.0, std=0.01)
        if self.coeff_predictor[-1].bias is not None:
            nn.init.zeros_(self.coeff_predictor[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C=d_model, H, W) -> output: (B, d_model, H, W)"""
        B, C, H, W = x.shape

        linear_out = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        if not self.use_gate:
            gate = torch.ones_like(gate)

        # Nonlinear features used for coefficient prediction
        x_nonlinear = F.gelu(self.nonlinear_proj(x))

        # Predict coefficients: shape -> (B, d_ff * total_coeffs, H, W)
        all_coeffs = self.coeff_predictor(x_nonlinear)
        # reshape to (B, d_ff, total_coeffs, H, W)
        all_coeffs = all_coeffs.view(B, self.d_ff, self.total_coeffs, H, W)

        # split poly / fourier coefficients
        poly_coeffs = all_coeffs[:, :, :self.num_poly_basis, :, :] if self.num_poly_basis > 0 else None
        fourier_coeffs = all_coeffs[:, :, self.num_poly_basis:, :, :] if self.num_fourier_basis > 0 else None

        # Create basis kernels on the device/dtype of input
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, self.kernel_size, device=x.device, dtype=x.dtype),
            indexing='ij'
        )

        poly_basis = self.poly_basis_2d(grid_x, grid_y, self.max_1d_degree).to(x.device) if self.num_poly_basis > 0 else None
        fourier_basis = self.fourier_basis_2d(grid_x, grid_y, self.max_1d_freq).to(x.device) if self.num_fourier_basis > 0 else None

        # Apply basis convolutions per-channel
        poly_out = self._apply_basis_convolution(x_nonlinear, poly_basis, poly_coeffs) if (poly_basis is not None) else torch.zeros_like(linear_out)
        fourier_out = self._apply_basis_convolution(x_nonlinear, fourier_basis, fourier_coeffs) if (fourier_basis is not None) else torch.zeros_like(linear_out)

        if not self.use_poly:
            poly_out = torch.zeros_like(poly_out)
        if not self.use_fourier:
            fourier_out = torch.zeros_like(fourier_out)

        poly_out = self.gamma_poly * poly_out
        fourier_out = self.gamma_fourier * fourier_out

        basis_mix = torch.sigmoid(self.basis_mix_logit)
        nonlinear_out = basis_mix * fourier_out + (1.0 - basis_mix) * poly_out

        linear_mix = torch.sigmoid(self.linear_mix_logit)
        combined_out = linear_mix * linear_out + (1.0 - linear_mix) * nonlinear_out

        gated_out = gate * combined_out
        return self.down_proj(self.dropout(gated_out))

    def _apply_basis_convolution(self, x: torch.Tensor, basis: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply a set of basis kernels to each channel separately, weighted by
        the predicted spatial coefficients.

        x: (B, d_ff, H, W)
        basis: (k, k, num_basis)
        coeffs: (B, d_ff, num_basis, H, W)
        returns: (B, d_ff, H, W)
        """
        if basis is None or coeffs is None:
            return torch.zeros_like(x)

        B, d_ff, H, W = x.shape
        num_basis = basis.shape[-1]
        k = basis.shape[0]

        # basis_kernels: (num_basis, 1, k, k)
        basis_kernels = basis.permute(2, 0, 1).unsqueeze(1).to(x.device).to(x.dtype)

        # reshape x to (B * d_ff, 1, H, W) so we can convolve with all basis kernels
        x_reshaped = x.contiguous().view(B * d_ff, 1, H, W)

        # conv produces (B * d_ff, num_basis, H, W)
        basis_applied = F.conv2d(x_reshaped, basis_kernels, padding=k // 2)

        # coeffs -> (B * d_ff, num_basis, H, W)
        coeffs_reshaped = coeffs.contiguous().view(B * d_ff, num_basis, H, W)

        # weighted sum over basis dimension
        out = (basis_applied * coeffs_reshaped).sum(dim=1)

        # reshape back to (B, d_ff, H, W)
        out = out.view(B, d_ff, H, W)
        return out

    def poly_basis_2d(self, x_grid: torch.Tensor, y_grid: torch.Tensor, max_degree: int) -> torch.Tensor:
        basis_list = []
        for dx in range(max_degree + 1):
            for dy in range(max_degree + 1):
                basis_list.append((x_grid ** dx) * (y_grid ** dy))
        if len(basis_list) == 0:
            return torch.empty(x_grid.shape + (0,), dtype=x_grid.dtype, device=x_grid.device)
        return torch.stack(basis_list, dim=-1)

    def fourier_basis_2d(self, x_grid: torch.Tensor, y_grid: torch.Tensor, max_freq: int) -> torch.Tensor:
        if max_freq == 0:
            return torch.empty(x_grid.shape + (0,), dtype=x_grid.dtype, device=x_grid.device)
        basis_list = []
        for freq in range(1, max_freq + 1):
            basis_list.extend([
                torch.cos(freq * math.pi * x_grid), torch.sin(freq * math.pi * x_grid),
                torch.cos(freq * math.pi * y_grid), torch.sin(freq * math.pi * y_grid)
            ])
        return torch.stack(basis_list, dim=-1)


# Module-level helpers so other classes can reuse the same basis builders
def poly_basis_2d(x_grid: torch.Tensor, y_grid: torch.Tensor, max_degree: int) -> torch.Tensor:
    basis_list = []
    for dx in range(max_degree + 1):
        for dy in range(max_degree + 1):
            basis_list.append((x_grid ** dx) * (y_grid ** dy))
    if len(basis_list) == 0:
        return torch.empty(x_grid.shape + (0,), dtype=x_grid.dtype, device=x_grid.device)
    return torch.stack(basis_list, dim=-1)


def fourier_basis_2d(x_grid: torch.Tensor, y_grid: torch.Tensor, max_freq: int) -> torch.Tensor:
    if max_freq == 0:
        return torch.empty(x_grid.shape + (0,), dtype=x_grid.dtype, device=x_grid.device)
    basis_list = []
    for freq in range(1, max_freq + 1):
        basis_list.extend([
            torch.cos(freq * math.pi * x_grid), torch.sin(freq * math.pi * x_grid),
            torch.cos(freq * math.pi * y_grid), torch.sin(freq * math.pi * y_grid)
        ])
    return torch.stack(basis_list, dim=-1)


def apply_basis_convolution(x: torch.Tensor, basis: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Module-level version of the class method for reuse.

    x: (B, d_ff, H, W)
    basis: (k, k, num_basis)
    coeffs: (B, d_ff, num_basis, H, W)
    returns: (B, d_ff, H, W)
    """
    if basis is None or coeffs is None:
        return torch.zeros_like(x)

    B, d_ff, H, W = x.shape
    num_basis = basis.shape[-1]
    k = basis.shape[0]

    basis_kernels = basis.permute(2, 0, 1).unsqueeze(1).to(x.device).to(x.dtype)
    x_reshaped = x.contiguous().view(B * d_ff, 1, H, W)
    basis_applied = F.conv2d(x_reshaped, basis_kernels, padding=k // 2)
    coeffs_reshaped = coeffs.contiguous().view(B * d_ff, num_basis, H, W)
    out = (basis_applied * coeffs_reshaped).sum(dim=1)
    out = out.view(B, d_ff, H, W)
    return out


class SpatioTemporalRevolutionaryGLU(nn.Module):
    """Router-blended function bank adapted for 2D spatio-temporal feature maps.

    This implements the RevolutionaryAdaPolyGLU idea but as a spatial module: it
    projects per-pixel features, evaluates a Chebyshev-style polynomial basis and
    a Fourier basis per-pixel, computes a GLU path, and dynamically routes/blends
    them at each spatial location using a small conv router.
    """
    def __init__(self, d_model: int, d_ff: int, spatial_kernel_size: int = 3,
                 max_degree: int = 3, n_fourier_terms: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.kernel_size = spatial_kernel_size
        self.max_degree = max_degree
        self.n_fourier_terms = n_fourier_terms

        # Projections
        self.proj_up = nn.Conv2d(d_model, d_ff, kernel_size=1)
        self.glu_proj = nn.Conv2d(d_model, d_ff * 2, kernel_size=1)
        self.down_proj = nn.Conv2d(d_ff, d_model, kernel_size=1)

        # Lightweight router (per-pixel logits for 3 bases)
        self.router = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, 3, kernel_size=1)
        )
        # start with gentler routing temperature so router doesn't make hard choices early
        self.temperature = nn.Parameter(torch.tensor(0.7))

        # Coefficient predictors are simple convs that output per-pixel coefficients
        self.cheby_coeff = nn.Sequential(
            nn.Conv2d(d_ff, d_ff, kernel_size=spatial_kernel_size, padding=spatial_kernel_size//2),
            nn.GELU(),
            nn.Conv2d(d_ff, d_ff * ((max_degree+1)**2), kernel_size=1)
        )
        self.fourier_coeff = nn.Sequential(
            nn.Conv2d(d_ff, d_ff, kernel_size=spatial_kernel_size, padding=spatial_kernel_size//2),
            nn.GELU(),
            nn.Conv2d(d_ff, d_ff * (4 * max(1, n_fourier_terms)), kernel_size=1)
        )

        # slightly stronger dropout for stability across seeds
        self.dropout = nn.Dropout2d(min(0.25, dropout + 0.05))
        # optional learnable fourier frequencies/amplitudes for adaptive bases
        self.use_learnable_fourier = (n_fourier_terms is not None and n_fourier_terms > 0)
        if self.use_learnable_fourier:
            self.fourier_freqs = nn.Parameter(torch.randn(n_fourier_terms) * 0.5)
            self.fourier_amps = nn.Parameter(torch.ones(n_fourier_terms) * 0.1)
        else:
            self.fourier_freqs = None
            self.fourier_amps = None

        # entropy regularization coeff for router (set to 0.0 by default)
        self.entropy_coeff = 0.0

        # training step counter for optional temperature annealing
        self.training_step = 0
        self._init_weights()

    def _init_weights(self):
        # Initialize convs with small normal std for smoother early behavior
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small additional init for router and coeff predictors if present
        try:
            for layer in self.router.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        except Exception:
            pass
        try:
            for layer in self.cheby_coeff.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        except Exception:
            pass
        try:
            for layer in self.fourier_coeff.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        except Exception:
            pass

    def router_entropy_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Encourage high entropy (diverse routing) by maximizing entropy.

        Returns a scalar loss (negative entropy) to add to training objective.
        """
        probs = F.softmax(router_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # (B, H, W)
        # mean over spatial and batch dims
        return -entropy.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=d_model, H, W)
        B, C, H, W = x.shape

        # Router logits and blend weights per-pixel
        router_logits = self.router(x)  # (B, 3, H, W)
        # Optionally anneal temperature slowly using training_step
        temp = float(self.temperature.item()) if isinstance(self.temperature, torch.Tensor) or isinstance(self.temperature, nn.Parameter) else float(self.temperature)
        if hasattr(self, 'training_step') and getattr(self, 'training_step', 0) > 0:
            temp = max(0.1, temp * (1.0 + 0.0005 * float(self.training_step)))
        blend = F.softmax(router_logits / max(0.1, temp), dim=1)  # (B,3,H,W)

        # GLU path
        glu = F.glu(self.glu_proj(x), dim=1)  # (B, d_ff, H, W)

        # Shared up projection for bases
        up = self.proj_up(x)  # (B, d_ff, H, W)

        # Chebyshev-like polynomial basis (use poly_basis_2d from above to generate kernels)
        # Predict per-pixel coefficients and apply basis convolution per-channel
        cheby_coeffs = self.cheby_coeff(up)
        num_cheby = (self.max_degree + 1) ** 2
        try:
            cheby_coeffs = cheby_coeffs.view(B, self.d_ff, num_cheby, H, W)
        except Exception:
            # fallback for deg=0
            cheby_coeffs = cheby_coeffs.view(B, self.d_ff, num_cheby, H, W)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, self.kernel_size, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        cheby_basis = poly_basis_2d(grid_x, grid_y, self.max_degree).to(x.device)  # (k,k,num_basis)
        cheby_out = apply_basis_convolution(up, cheby_basis, cheby_coeffs) if cheby_basis.shape[-1] > 0 else torch.zeros_like(up)

        # Fourier path
        fourier_coeffs = self.fourier_coeff(up)
        num_fourier = 4 * max(1, self.n_fourier_terms)
        fourier_coeffs = fourier_coeffs.view(B, self.d_ff, num_fourier, H, W)
        # Optionally modulate fourier basis using learnable freqs/amps
        if self.use_learnable_fourier and self.fourier_freqs is not None:
            # construct a simple Fourier basis using learned freqs and amplitudes
            fb_list = []
            for i in range(len(self.fourier_freqs)):
                f = float(self.fourier_freqs[i].item())
                a = float(self.fourier_amps[i].item())
                fb_list.extend([
                    a * torch.cos(f * math.pi * grid_x), a * torch.sin(f * math.pi * grid_x),
                    a * torch.cos(f * math.pi * grid_y), a * torch.sin(f * math.pi * grid_y)
                ])
            fourier_basis = torch.stack(fb_list, dim=-1).to(x.device)
        else:
            fourier_basis = fourier_basis_2d(grid_x, grid_y, self.n_fourier_terms).to(x.device)
        fourier_out = apply_basis_convolution(up, fourier_basis, fourier_coeffs) if fourier_basis.shape[-1] > 0 else torch.zeros_like(up)

        # Stack paths and blend per-pixel
        # shapes: (B, d_ff, H, W)
        stacked = torch.stack([glu, cheby_out, fourier_out], dim=1)  # (B,3,d_ff,H,W)
        blend_u = blend.unsqueeze(2)  # (B,3,1,H,W)
        mixed = (stacked * blend_u).sum(dim=1)  # (B, d_ff, H, W)

        return self.down_proj(self.dropout(mixed))


class SpatioTemporalRoutedGLU(nn.Module):
    """Simple spatial router that blends a linear conv path with the adaptive
    SpatioTemporalFourierAdaPolyGLU path per-pixel. This provides a lightweight
    routed variant for comparison.
    """
    def __init__(self, d_model: int, d_ff: int, spatial_kernel_size: int = 3,
                 max_1d_degree: int = 2, max_1d_freq: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # adaptive path (reuses the Fourier+poly GLU)
        self.adaptive = SpatioTemporalFourierAdaPolyGLU(d_model, d_ff,
                                                       spatial_kernel_size=spatial_kernel_size,
                                                       max_1d_degree=max_1d_degree,
                                                       max_1d_freq=max_1d_freq)

        # linear path
        self.linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU()
        )

        # router: per-pixel blending weight in [0,1]
        self.router = nn.Conv2d(d_model, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, H, W)
        adapt = self.adaptive(x)
        lin = self.linear(x)
        w = torch.sigmoid(self.router(x))  # (B,1,H,W)
        out = w * adapt + (1.0 - w) * lin
        return self.dropout(out)


# =============================================================================
# 2. State-of-the-Art PredRNN++ Implementation
# =============================================================================
# This is a simplified, yet functional, implementation of the core PredRNN++
# components for a fair comparison. It uses the same ConvLSTMCell.

class ST_LSTM_Cell(nn.Module):
    """
    Spatio-Temporal LSTM Cell from PredRNN++
    """
    def __init__(self, in_channels, num_hidden, kernel_size=5):
        super().__init__()
        self.num_hidden = num_hidden
        padding = kernel_size // 2

        self.conv_x_h = nn.Conv2d(in_channels + num_hidden, num_hidden * 4, kernel_size=kernel_size, padding=padding)
        self.conv_x_s = nn.Conv2d(in_channels + num_hidden, num_hidden * 4, kernel_size=kernel_size, padding=padding)
        self.conv_s_h = nn.Conv2d(num_hidden + num_hidden, num_hidden * 4, kernel_size=kernel_size, padding=padding)

    def forward(self, x, h, c, s):
        x_plus_h = torch.cat([x, h], dim=1)
        x_plus_s = torch.cat([x, s], dim=1)
        s_plus_h = torch.cat([s, h], dim=1)

        gates_xh = self.conv_x_h(x_plus_h)
        gates_xs = self.conv_x_s(x_plus_s)
        gates_sh = self.conv_s_h(s_plus_h)

        i_x, f_x, g_x, o_x = torch.split(gates_xh, self.num_hidden, dim=1)
        i_s, f_s, g_s, o_s = torch.split(gates_xs, self.num_hidden, dim=1)
        i_sh, f_sh, g_sh, o_sh = torch.split(gates_sh, self.num_hidden, dim=1)

        i = torch.sigmoid(i_x + i_s + i_sh)
        f = torch.sigmoid(f_x + f_s + f_sh)
        g = torch.tanh(g_x + g_s + g_sh)
        o = torch.sigmoid(o_x + o_s + o_sh)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, s

class PredRNNpp(nn.Module):
    def __init__(self, in_channels=1, num_hidden=32, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        # The first cell takes the input image as a layer input
        self.cells.append(ST_LSTM_Cell(in_channels, num_hidden))
        # Subsequent cells take the hidden state of the previous layer as input
        for _ in range(self.num_layers - 1):
            self.cells.append(ST_LSTM_Cell(num_hidden, num_hidden))

        # Use Sigmoid to produce outputs in [0,1], matching dataset range
        self.decoder = nn.Sequential(
            nn.Conv2d(num_hidden, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, future_steps=10):
        B, T, C, H, W = input_tensor.shape
        # create separate tuples for each layer (avoid repeated reference)
        hidden_states = [(None, None, None) for _ in range(self.num_layers)]

        # Encoding phase: Process input frames one by one
        for t in range(T):
            # The input for the first layer is the image at this time step
            h_from_prev_layer = input_tensor[:, t, :, :, :]

            # Pass through all layers sequentially
            for i in range(self.num_layers):
                h, c, s = hidden_states[i]
                if h is None:
                    h = torch.zeros(B, self.cells[i].num_hidden, H, W, device=device)
                    c = torch.zeros(B, self.cells[i].num_hidden, H, W, device=device)
                    s = torch.zeros(B, self.cells[i].num_hidden, H, W, device=device)

                h_next, c_next, s_next = self.cells[i](h_from_prev_layer, h, c, s)
                hidden_states[i] = (h_next, c_next, s_next)

                # The output of the current layer becomes the input for the next layer
                h_from_prev_layer = h_next

        # Decoding phase: Generate future frames autoregressively
        predictions = []
        # The first input to the decoding phase is the last hidden state from encoding
        # This acts as the "initial image" for the first time step of prediction.
        predicted_image = self.decoder(hidden_states[-1][0])
        predictions.append(predicted_image)

        for t in range(1, future_steps):
            # Use the previously predicted image as the input for the first layer
            h_from_prev_layer = predicted_image

            # Pass through all layers, updating the hidden states
            for i in range(self.num_layers):
                h, c, s = hidden_states[i]
                h_next, c_next, s_next = self.cells[i](h_from_prev_layer, h, c, s)
                hidden_states[i] = (h_next, c_next, s_next)
                h_from_prev_layer = h_next

            # Decode the final hidden state to get the next predicted image
            predicted_image = self.decoder(hidden_states[-1][0])
            predictions.append(predicted_image)

        return torch.stack(predictions, dim=1)

# =============================================================================
# 3. Enhanced PredRNN++ with STFAGLU
# =============================================================================
# This model uses the same PredRNN++ structure but replaces the final
# convolutional decoder with your custom STFAGLU module for a fair comparison.

class STFAGLU_PredRNNpp(nn.Module):
    def __init__(self, in_channels=1, num_hidden=32, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.cells.append(ST_LSTM_Cell(in_channels, num_hidden))
        for _ in range(self.num_layers - 1):
            self.cells.append(ST_LSTM_Cell(num_hidden, num_hidden))

        self.decoder = nn.Sequential(
            SpatioTemporalFourierAdaPolyGLU(num_hidden, num_hidden,
                                            spatial_kernel_size=3, max_1d_degree=2, max_1d_freq=1),
            nn.Conv2d(num_hidden, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, future_steps=10):
        B, T, C, H, W = input_tensor.shape
        # create separate tuples for each layer (avoid repeated reference)
        hidden_states = [(None, None, None) for _ in range(self.num_layers)]

        # Encoding phase: Process input frames one by one
        for t in range(T):
            h_from_prev_layer = input_tensor[:, t, :, :, :]

            for i in range(self.num_layers):
                h, c, s = hidden_states[i]
                if h is None:
                    h = torch.zeros(B, self.cells[i].num_hidden, H, W, device=device)
                    c = torch.zeros(B, self.cells[i].num_hidden, H, W, device=device)
                    s = torch.zeros(B, self.cells[i].num_hidden, H, W, device=device)

                h_next, c_next, s_next = self.cells[i](h_from_prev_layer, h, c, s)
                hidden_states[i] = (h_next, c_next, s_next)

                h_from_prev_layer = h_next

        # Decoding phase: Generate future frames autoregressively
        predictions = []
        predicted_image = self.decoder(hidden_states[-1][0])
        predictions.append(predicted_image)

        for t in range(1, future_steps):
            h_from_prev_layer = predicted_image

            for i in range(self.num_layers):
                h, c, s = hidden_states[i]
                h_next, c_next, s_next = self.cells[i](h_from_prev_layer, h, c, s)
                hidden_states[i] = (h_next, c_next, s_next)
                h_from_prev_layer = h_next

            predicted_image = self.decoder(hidden_states[-1][0])
            predictions.append(predicted_image)

        return torch.stack(predictions, dim=1)

# =============================================================================
# 4. Moving MNIST Dataset
# =============================================================================

class MovingMNISTDataset(Dataset):
    def __init__(self, num_samples=1200, num_frames=20, image_size=32, num_digits=2):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_digits = num_digits
        self.sequences = self._generate_sequences()

    def _generate_sequences(self):
        sequences = []
        for _ in range(self.num_samples):
            sequence = []
            digits = []
            for _ in range(self.num_digits):
                pos = np.random.rand(2) * 0.8 + 0.1
                velocity = np.random.randn(2) * 0.08 + 0.02
                digits.append({'pos': pos, 'velocity': velocity})

            for _ in range(self.num_frames):
                frame = torch.zeros((self.image_size, self.image_size))

                for digit in digits:
                    digit['pos'] += digit['velocity']

                    for j in range(2):
                        if digit['pos'][j] < 0.05:
                            digit['pos'][j] = 0.05
                            digit['velocity'][j] = abs(digit['velocity'][j]) * (0.9 + 0.2 * np.random.random())
                        elif digit['pos'][j] > 0.95:
                            digit['pos'][j] = 0.95
                            digit['velocity'][j] = -abs(digit['velocity'][j]) * (0.9 + 0.2 * np.random.random())

                    x, y = (digit['pos'] * self.image_size).astype(int)
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                                    dist = math.sqrt(dx**2 + dy**2)
                                    intensity = max(0, 1.0 - dist/2.0)
                                    frame[nx, ny] = max(frame[nx, ny], intensity)

                sequence.append(frame.unsqueeze(0))
            sequences.append(torch.stack(sequence))
        return sequences

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return sequence[:10], sequence[10:]

# =============================================================================
# 5. Training and Evaluation
# =============================================================================

def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, model_name="Model", weight_decay=1e-5, coeff_lr_mult: float = 5.0, use_amp: bool = False, clip_norm: float = 1.0, router_entropy_weight: float = 0.0, fallback_to_cpu_on_oom: bool = True):
    """Train the model. If a CUDA OOM occurs and `fallback_to_cpu_on_oom` is True,
    this will attempt to retry training on CPU with a reduced batch size.
    Returns: train_losses, val_losses, total_time, best_val_loss
    """
    model = model.to(device)
    
    # Effective AMP usage depends on availability
    use_amp_effective = use_amp and torch.cuda.is_available()
    if use_amp_effective:
        print(f"AMP enabled for {model_name}")
    
    # Create GradScaler for mixed precision (new torch.amp API)
    # Use device_type='cuda' when AMP is enabled on CUDA
    # GradScaler signature varies across torch versions; use the compatible form
    scaler = torch.amp.GradScaler(enabled=use_amp_effective)

    # Create optimizer with special parameter group for STFAGLU coefficient predictor
    # so that coeff_predictor parameters can learn faster and are not overly regularized.
    coeff_params = []
    for m in model.modules():
        if isinstance(m, SpatioTemporalFourierAdaPolyGLU):
            # collect coeff_predictor params if present
            if hasattr(m, 'coeff_predictor'):
                coeff_params += [p for p in m.coeff_predictor.parameters() if p.requires_grad]

    # remove duplicates
    coeff_param_ids = {id(p) for p in coeff_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in coeff_param_ids]

    param_groups = []
    if coeff_params:
        # Base group: all other params use the provided lr and weight decay
        param_groups.append({'params': other_params, 'lr': lr, 'weight_decay': weight_decay})
        # Coeff predictor: higher LR, no weight decay
        param_groups.append({'params': coeff_params, 'lr': lr * coeff_lr_mult, 'weight_decay': 0.0})
    else:
        param_groups.append({'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay})

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    start_time = time.time()
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Use autocast for the forward pass (new torch.amp API)
            with torch.amp.autocast('cuda', enabled=use_amp_effective):
                outputs = model(inputs, future_steps=targets.size(1))
                loss = criterion(outputs, targets)

            # Optionally add router entropy regularizer (if any rev blocks expose last_router_logits)
            if router_entropy_weight and router_entropy_weight > 0.0:
                ent_loss = 0.0
                ent_count = 0
                for m in model.modules():
                    # look for stored logits (v2) or rev block with last_router_logits
                    if hasattr(m, 'last_router_logits'):
                        logits = m.last_router_logits  # (B*T, n_bases, H, W)
                        probs = F.softmax(logits.view(logits.shape[0], logits.shape[1], -1), dim=1)
                        ent = - (probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                        ent_loss = ent_loss + ent
                        ent_count += 1
                if ent_count > 0:
                    ent_loss = ent_loss / float(ent_count)
                    loss = loss + router_entropy_weight * ent_loss

            # Scale loss and call backward
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            if clip_norm is not None and clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            # Scaler step
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        avg_val_loss = evaluate_model(model, val_loader)
        val_losses.append(avg_val_loss)

        # Extra diagnostics: compute per-batch validation losses to inspect small changes
        try:
            batch_losses = []
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp_effective):
                        v_outputs = model(v_inputs, future_steps=v_targets.size(1))
                    # per-batch MSE
                    b_loss = float(torch.mean((v_outputs - v_targets) ** 2).item())
                    batch_losses.append(b_loss)
            if batch_losses:
                bmin = float(np.min(batch_losses))
                bmax = float(np.max(batch_losses))
                bstd = float(np.std(batch_losses))
                # Print with higher precision to reveal tiny changes
                print(f"{model_name} - Val batch loss min/mean/max/std: {bmin:.9f} / {np.mean(batch_losses):.9f} / {bmax:.9f} / {bstd:.9f}")
        except Exception:
            pass

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # save checkpoint
            try:
                ckpt_path = f"{model_name}_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss
                }, ckpt_path)
            except Exception:
                pass

        scheduler.step()

        # Print epoch summary every epoch (helps debugging when val loss looks constant)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}')

            # Debug: inspect a single validation batch's output/target statistics to catch identical predictions
        try:
            model.eval()
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp_effective):
                        v_outputs = model(v_inputs, future_steps=v_targets.size(1))
                    # print means to detect pathological constant outputs
                    print(f"[DEBUG] {model_name} val sample - out_mean={v_outputs.mean().item():.6f}, tgt_mean={v_targets.mean().item():.6f}")
                    break
        except Exception:
            pass

        end_time = time.time()
        total_time = end_time - start_time
        print(f"{model_name} - Training finished in {total_time:.2f} seconds. Best Validation Loss: {best_val_loss:.6f}")
        return train_losses, val_losses, total_time, best_val_loss
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg and fallback_to_cpu_on_oom and torch.cuda.is_available():
            # Try to recover and retry on CPU with a much smaller batch size
            print(f"CUDA out of memory during training of {model_name}: {e}")
            print("Attempting graceful fallback: emptying cache and retrying on CPU with smaller batch size.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # suggest environment tweak for fragmentation
            print("If fragmentation persists consider setting: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

            # Determine original batch size if available
            try:
                orig_batch = int(getattr(train_loader, 'batch_size', 1) or 1)
            except Exception:
                orig_batch = 1
            new_batch = max(1, orig_batch // 4)
            if new_batch >= orig_batch:
                new_batch = 1

            print(f"Retrying on CPU with batch_size={new_batch} (original {orig_batch})")

            # Build reduced loaders using same dataset objects where possible
            try:
                new_train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=new_batch, shuffle=True, num_workers=0, collate_fn=getattr(train_loader, 'collate_fn', None))
                new_val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=new_batch, shuffle=False, num_workers=0, collate_fn=getattr(val_loader, 'collate_fn', None))
            except Exception as e_newldr:
                print('Could not construct reduced DataLoaders for fallback:', e_newldr)
                raise

            # Move global device to CPU so module code that uses the module-level `device` variable will behave correctly
            try:
                globals()['device'] = torch.device('cpu')
            except Exception:
                pass

            # Move model to CPU and retry training with AMP disabled
            try:
                model = model.to(torch.device('cpu'))
            except Exception:
                pass

            # Recursively call train_model with reduced loaders and fallback turned off to avoid loops
            return train_model(model, new_train_loader, new_val_loader, num_epochs=num_epochs, lr=lr, model_name=model_name, weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=False, clip_norm=clip_norm, router_entropy_weight=router_entropy_weight, fallback_to_cpu_on_oom=False)
        # otherwise re-raise
        raise


def psnr_from_mse(mse, max_val=1.0):
    # mse can be array-like
    mse = np.array(mse)
    with np.errstate(divide='ignore'):
        psnr = 10 * np.log10((max_val ** 2) / mse)
    psnr[np.isinf(psnr)] = 100.0
    return psnr


def evaluate_model_metrics(model, test_loader):
    """Evaluate per-timestep MSE and PSNR on test_loader. Returns (mse_per_timestep, psnr_per_timestep)."""
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    sum_mse = None
    count = 0
    sum_ssim = None
    # optional LPIPS (dynamically imported to avoid static lint errors when not installed)
    lpips_model = None
    try:
        import importlib, importlib.util
        if importlib.util.find_spec('lpips') is not None:
            lpips = importlib.import_module('lpips')
            lpips_model = lpips.LPIPS(net='alex').to(device)
        else:
            lpips_model = None
    except Exception:
        lpips_model = None

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Use autocast during evaluation when CUDA is available (new torch.amp API)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs, future_steps=targets.size(1))
            # ensure outputs are in a valid image range
            outputs = outputs.clamp(0.0, 1.0)
            # outputs, targets shape: (B, T, C, H, W)
            per_example_mse = torch.mean((outputs - targets) ** 2, dim=[2, 3, 4])  # (B, T)
            per_batch_mse = torch.sum(per_example_mse, dim=0).cpu().numpy()  # (T,)
            if sum_mse is None:
                sum_mse = per_batch_mse
            else:
                sum_mse += per_batch_mse

            # SSIM per-frame (average over batch and channels) — basic PyTorch implementation
            try:
                # compute SSIM per timestep using a simple windowless formula (mean-based approx)
                B, T, C, H, W = outputs.shape
                outputs_flat = outputs.view(B * T, C, H, W)
                targets_flat = targets.view(B * T, C, H, W)
                mu_x = outputs_flat.mean(dim=[2, 3], keepdim=True)
                mu_y = targets_flat.mean(dim=[2, 3], keepdim=True)
                sigma_x = ((outputs_flat - mu_x) ** 2).mean(dim=[2, 3], keepdim=True)
                sigma_y = ((targets_flat - mu_y) ** 2).mean(dim=[2, 3], keepdim=True)
                sigma_xy = ((outputs_flat - mu_x) * (targets_flat - mu_y)).mean(dim=[2, 3], keepdim=True)
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2
                ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
                ssim_per_pair = ssim_map.view(B, T, C, 1, 1).mean(dim=[2, 3, 4]).cpu().numpy()  # (B, T)
                per_batch_ssim = ssim_per_pair.sum(axis=0)  # (T,)
                if sum_ssim is None:
                    sum_ssim = per_batch_ssim
                else:
                    sum_ssim += per_batch_ssim
            except Exception:
                sum_ssim = None

            # LPIPS (if available) averaged per timestep
            if lpips_model is not None:
                # LPIPS expects 3-channel images in [-1,1]; tile single-channel to 3
                B, T, C, H, W = outputs.shape
                out_lp = outputs.repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W) * 2.0 - 1.0
                tgt_lp = targets.repeat(1, 1, 3, 1, 1).view(B * T, 3, H, W) * 2.0 - 1.0
                lpips_vals = lpips_model(out_lp, tgt_lp).view(B, T).cpu().numpy()  # (B,T)
                # store in sum_mse as additional side-channel if desired (we'll return separately)
                if 'sum_lpips' not in locals():
                    sum_lpips = lpips_vals.sum(axis=0)
                else:
                    sum_lpips += lpips_vals.sum(axis=0)

            count += inputs.size(0)

    mse_per_timestep = sum_mse / count
    psnr_per_timestep = psnr_from_mse(mse_per_timestep, max_val=1.0)
    ssim_per_timestep = (sum_ssim / count) if (sum_ssim is not None and count > 0) else None
    lpips_per_timestep = (sum_lpips / count) if ('sum_lpips' in locals() and count > 0) else None
    return mse_per_timestep, psnr_per_timestep, ssim_per_timestep, lpips_per_timestep

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs, future_steps=targets.size(1))
            outputs = outputs.clamp(0.0, 1.0)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


# ---------------------
# Ablation helpers
# ---------------------
def make_stfaglu_predrnn(in_channels=1, num_hidden=32, num_layers=2,
                         use_poly=True, use_fourier=True, use_gate=True,
                         max_deg=2, max_freq=1):
    """Create a STFAGLU_PredRNNpp model with a configured STFAGLU decoder."""
    model = STFAGLU_PredRNNpp(in_channels=in_channels, num_hidden=num_hidden, num_layers=num_layers)
    model.decoder = nn.Sequential(
        SpatioTemporalFourierAdaPolyGLU(num_hidden, num_hidden,
                                        spatial_kernel_size=3,
                                        max_1d_degree=max_deg,
                                        max_1d_freq=max_freq,
                                        use_poly=use_poly,
                                        use_fourier=use_fourier,
                                        use_gate=use_gate),
        nn.Conv2d(num_hidden, 1, 1),
        nn.Sigmoid()
    )
    return model


def ablation_experiments(train_loader, val_loader, test_loader, quick_epochs=5):
    """Run a small set of ablation experiments and return results dict.
    This runs quick short training runs (quick_epochs) for quick attribution signals.
    Increase quick_epochs to get stronger, more reliable signals.
    """
    experiments = []
    # baseline (full model)
    experiments.append(("baseline",  True,  True,  True, 2, 1))
    # remove components
    experiments.append(("no_poly",   False, True,  True, 2, 1))
    experiments.append(("no_fourier",True,  False, True, 2, 1))
    experiments.append(("no_gate",   True,  True,  False,2, 1))
    # degree/frequency sweep (small grid)
    experiments.append(("deg0_freq0", True, True, True, 0, 0))
    experiments.append(("deg1_freq0", True, True, True, 1, 0))
    experiments.append(("deg2_freq1", True, True, True, 2, 1))

    results = {}
    for name, use_poly, use_fourier, use_gate, deg, freq in experiments:
        print(f"\nRunning ablation: {name} (poly={use_poly}, fourier={use_fourier}, gate={use_gate}, deg={deg}, freq={freq})")
        model = make_stfaglu_predrnn(in_channels=1, num_hidden=32, num_layers=2,
                                     use_poly=use_poly, use_fourier=use_fourier, use_gate=use_gate,
                                     max_deg=deg, max_freq=freq)
        try:
            # Use default coeff multiplier here (can be modified if needed)
            train_model(model, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"ablation_{name}")
            test_loss = evaluate_model(model, test_loader)
            mse_ts, psnr_ts, ssim_ts, lpips_ts = evaluate_model_metrics(model, test_loader)
            results[name] = {'test_loss': test_loss, 'mse_ts': mse_ts, 'psnr_ts': psnr_ts}
            print(f"Ablation {name} -> Test Loss: {test_loss:.6f}, PSNR t1: {psnr_ts[0]:.2f} dB")
        except Exception as e:
            print(f"Ablation {name} failed: {e}")
            results[name] = {'test_loss': float('inf'), 'mse_ts': None, 'psnr_ts': None}

    return results

# =============================================================================
# 6. Main Experiment and Visualization
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='STFAGLU vs PredRNN++ experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--amp', action='store_true', help='Enable AMP (automatic mixed precision)')
    parser.add_argument('--multi-seed', action='store_true', help='Run multi-seed experiments and save CSV')
    parser.add_argument('--seeds', type=str, default='42,7,123', help='Comma-separated seeds for multi-seed experiments')
    parser.add_argument('--coeff-lr-mult', type=float, default=5.0, help='Multiplier for coeff predictor LR')
    parser.add_argument('--grid', action='store_true', help='Run a small grid search over coeff_lr_mult x weight_decay')
    parser.add_argument('--degree-sweep', type=str, default='', help='Comma-separated degrees to sweep (e.g. 0,1,2,3)')
    parser.add_argument('--num-samples', type=int, default=1200, help='Number of MovingMNIST sequences to generate for dataset')
    # When running inside a notebook (Kaggle/Colab), kernel args are injected; tolerate them
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # apply seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("PUBLICATION-READY COMPARISON: 3 STF VARIANTS (Original, Routed, Revolutionary) vs PredRNN++")
    print("=" * 60)

    print(f"Creating enhanced Moving MNIST dataset (2-digit) with {args.num_samples} samples...")
    dataset = MovingMNISTDataset(num_samples=args.num_samples, num_frames=20, image_size=32, num_digits=2)

    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Choose num_workers: 0 on Windows; use a small pool on Linux (Kaggle) to speed loading
    if os.name == 'nt':
        num_workers = 0
    else:
        num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    def run_multi_seed_experiments(seeds_list, quick_epochs=6, coeff_lr_mult=5.0, weight_decay=1e-6):
        rows = []
        for sd in seeds_list:
            print(f"\n--- Multi-seed run: seed={sd} ---")
            torch.manual_seed(sd)
            np.random.seed(sd)
            random.seed(sd)

            # Recreate datasets and loaders for each seed
            dataset = MovingMNISTDataset(num_samples=args.num_samples, num_frames=20, image_size=32, num_digits=2)
            train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.4, random_state=sd)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=sd)
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

            # Baseline
            predrnn_model = PredRNNpp(in_channels=1, num_hidden=32, num_layers=2)
            t0 = time.time()
            train_model(predrnn_model, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"PredRNN_seed{sd}", weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=args.amp)
            predrnn_time = time.time() - t0
            predrnn_test_loss = evaluate_model(predrnn_model, test_loader)
            predrnn_mse_ts, predrnn_psnr_ts, predrnn_ssim_ts, predrnn_lpips_ts = evaluate_model_metrics(predrnn_model, test_loader)

            # STF variants: original (SpatioTemporalFourierAdaPolyGLU), routed, revolutionary
            # 1) Original STFAGLU
            stf_orig = make_stfaglu_predrnn(in_channels=1, num_hidden=32, num_layers=2)
            t0 = time.time()
            train_model(stf_orig, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"STF_orig_seed{sd}", weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=args.amp)
            stf_orig_time = time.time() - t0
            stf_orig_test_loss = evaluate_model(stf_orig, test_loader)
            stf_orig_mse_ts, stf_orig_psnr_ts, stf_orig_ssim_ts, stf_orig_lpips_ts = evaluate_model_metrics(stf_orig, test_loader)

            # 2) Routed variant
            stf_routed = STFAGLU_PredRNNpp(in_channels=1, num_hidden=32, num_layers=2)
            # replace decoder with routed block
            stf_routed.decoder = nn.Sequential(
                SpatioTemporalRoutedGLU(32, 32, spatial_kernel_size=3, max_1d_degree=2, max_1d_freq=1),
                nn.Conv2d(32, 1, 1), nn.Sigmoid()
            )
            t0 = time.time()
            train_model(stf_routed, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"STF_routed_seed{sd}", weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=args.amp)
            stf_routed_time = time.time() - t0
            stf_routed_test_loss = evaluate_model(stf_routed, test_loader)
            stf_routed_mse_ts, stf_routed_psnr_ts, stf_routed_ssim_ts, stf_routed_lpips_ts = evaluate_model_metrics(stf_routed, test_loader)

            # 3) Revolutionary variant
            stf_rev = STFAGLU_PredRNNpp(in_channels=1, num_hidden=32, num_layers=2)
            stf_rev.decoder = nn.Sequential(
                SpatioTemporalRevolutionaryGLU(32, 32, spatial_kernel_size=3, max_degree=3, n_fourier_terms=3),
                nn.Conv2d(32, 1, 1), nn.Sigmoid()
            )
            t0 = time.time()
            train_model(stf_rev, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"STF_rev_seed{sd}", weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=args.amp)
            stf_rev_time = time.time() - t0
            stf_rev_test_loss = evaluate_model(stf_rev, test_loader)
            stf_rev_mse_ts, stf_rev_psnr_ts, stf_rev_ssim_ts, stf_rev_lpips_ts = evaluate_model_metrics(stf_rev, test_loader)

            # Diagnostic: compare model outputs on a single test batch to see if models produce identical predictions
            try:
                model_list = [('predrnn', predrnn_model), ('stf_orig', stf_orig), ('stf_routed', stf_routed), ('stf_rev', stf_rev)]
                with torch.no_grad():
                    for batch_inputs, batch_targets in test_loader:
                        batch_inputs = batch_inputs.to(device)
                        batch_targets = batch_targets.to(device)
                        outputs = {}
                        for name, m in model_list:
                            m.eval()
                            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                                out = m(batch_inputs, future_steps=batch_targets.size(1))
                            outputs[name] = out.detach()
                        break

                # compute pairwise mean absolute differences
                names = list(outputs.keys())
                for i in range(len(names)):
                    for j in range(i+1, len(names)):
                        a = outputs[names[i]]
                        b = outputs[names[j]]
                        mad = torch.mean(torch.abs(a - b)).item()
                        print(f"[DIAG] mean abs diff {names[i]} vs {names[j]}: {mad:.6e}")
            except Exception:
                pass

            # Learned params for original STFAGLU (if present)
            gamma_poly = gamma_fourier = basis_mix = linear_mix = None
            for m in stf_orig.modules():
                if isinstance(m, SpatioTemporalFourierAdaPolyGLU):
                    try:
                        gamma_poly = float(m.gamma_poly.detach().cpu().item())
                        gamma_fourier = float(m.gamma_fourier.detach().cpu().item())
                        basis_mix = float(torch.sigmoid(m.basis_mix_logit).detach().cpu().item())
                        linear_mix = float(torch.sigmoid(m.linear_mix_logit).detach().cpu().item())
                    except Exception:
                        pass
                    break

            rows.append({
                'seed': sd,
                'predrnn_test_loss': float(predrnn_test_loss),
                'stf_orig_test_loss': float(stf_orig_test_loss),
                'stf_routed_test_loss': float(stf_routed_test_loss),
                'stf_rev_test_loss': float(stf_rev_test_loss),
                'predrnn_psnr_t1': float(predrnn_psnr_ts[0]),
                'stf_orig_psnr_t1': float(stf_orig_psnr_ts[0]),
                'stf_routed_psnr_t1': float(stf_routed_psnr_ts[0]),
                'stf_rev_psnr_t1': float(stf_rev_psnr_ts[0]),
                'predrnn_time': float(predrnn_time),
                'stf_orig_time': float(stf_orig_time),
                'stf_routed_time': float(stf_routed_time),
                'stf_rev_time': float(stf_rev_time),
                'gamma_poly': gamma_poly,
                'gamma_fourier': gamma_fourier,
                'basis_mix': basis_mix,
                'linear_mix': linear_mix,
                'predrnn_mse_ts': json.dumps(predrnn_mse_ts.tolist()),
                'stf_orig_mse_ts': json.dumps(stf_orig_mse_ts.tolist()),
                'stf_routed_mse_ts': json.dumps(stf_routed_mse_ts.tolist()),
                'stf_rev_mse_ts': json.dumps(stf_rev_mse_ts.tolist()),
                'predrnn_psnr_ts': json.dumps(predrnn_psnr_ts.tolist()),
                'stf_orig_psnr_ts': json.dumps(stf_orig_psnr_ts.tolist()),
                'stf_routed_psnr_ts': json.dumps(stf_routed_psnr_ts.tolist()),
                'stf_rev_psnr_ts': json.dumps(stf_rev_psnr_ts.tolist())
            })

        # Save CSV into the same directory as this script so it's easy to find
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            script_dir = os.getcwd()
        csv_path = os.path.join(script_dir, 'multi_seed_results.csv')
        keys = rows[0].keys() if rows else []
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(keys))
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Multi-seed results saved to {csv_path}")
        except Exception as e:
            print('Failed to write multi-seed CSV:', e)

        # Print aggregated summary including all STF variants and per-seed relative improvements
        pred_losses = np.array([r['predrnn_test_loss'] for r in rows])
        stf_orig_losses = np.array([r.get('stf_orig_test_loss', np.nan) for r in rows])
        stf_routed_losses = np.array([r.get('stf_routed_test_loss', np.nan) for r in rows])
        stf_rev_losses = np.array([r.get('stf_rev_test_loss', np.nan) for r in rows])

        # Print mean/std for each model's test loss
        print('\nMulti-seed summary:')
        print(f'PredRNN++ test loss mean/std: {pred_losses.mean():.6f} / {pred_losses.std():.6f}')
        print(f'STFAGLU test loss mean/std:   {stf_orig_losses.mean():.6f} / {stf_orig_losses.std():.6f}')
        print(f'STF Routed test loss mean/std:{stf_routed_losses.mean():.6f} / {stf_routed_losses.std():.6f}')
        print(f'STF Rev test loss mean/std:   {stf_rev_losses.mean():.6f} / {stf_rev_losses.std():.6f}')

        # Compute per-seed relative improvements (per-row) and report mean ± std
        # relative improvement = (Pred - STF) / Pred * 100
        def rel_improvement(pred_arr, stf_arr):
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = (pred_arr - stf_arr) / pred_arr * 100.0
                rel = rel[~np.isnan(rel)]
            return rel

        rel_orig = rel_improvement(pred_losses, stf_orig_losses)
        rel_routed = rel_improvement(pred_losses, stf_routed_losses)
        rel_rev = rel_improvement(pred_losses, stf_rev_losses)

        if rel_orig.size > 0:
            print(f'Mean relative improvement (orig): {rel_orig.mean():.3f}% ± {rel_orig.std():.3f}%')
        if rel_routed.size > 0:
            print(f'Mean relative improvement (routed): {rel_routed.mean():.3f}% ± {rel_routed.std():.3f}%')
        if rel_rev.size > 0:
            print(f'Mean relative improvement (rev): {rel_rev.mean():.3f}% ± {rel_rev.std():.3f}%')

        return rows

    def run_grid_search(seeds_list, quick_epochs=8, coeff_grid=[3.0, 5.0, 10.0], wd_grid=[1e-6, 1e-5]):
        all_rows = []
        for coeff in coeff_grid:
            for wd in wd_grid:
                print(f"\n=== Grid run: coeff_lr_mult={coeff}, weight_decay={wd} ===")
                rows = run_multi_seed_experiments(seeds_list, quick_epochs=quick_epochs, coeff_lr_mult=coeff, weight_decay=wd)
                # aggregate
                pred_losses = np.array([r['predrnn_test_loss'] for r in rows])
                stf_losses = np.array([r.get('stf_orig_test_loss', np.nan) for r in rows])
                all_rows.append({'coeff_lr_mult': coeff, 'weight_decay': wd,
                                 'pred_mean': float(pred_losses.mean()), 'pred_std': float(pred_losses.std()),
                                 'stf_mean': float(stf_losses.mean()), 'stf_std': float(stf_losses.std()),
                                 'mean_rel_impr_pct': float((pred_losses.mean() - stf_losses.mean())/pred_losses.mean()*100.0)})

        # Save grid CSV into the same directory as this script
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            script_dir = os.getcwd()
        grid_csv = os.path.join(script_dir, 'grid_results.csv')
        keys = all_rows[0].keys() if all_rows else []
        try:
            with open(grid_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(keys))
                writer.writeheader()
                for r in all_rows:
                    writer.writerow(r)
            print(f"Grid results saved to {grid_csv}")
        except Exception as e:
            print('Failed to write grid CSV:', e)
        return all_rows

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # If user requested grid search, run and exit early
    if args.grid:
        try:
            seeds_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except Exception:
            seeds_list = [42, 7, 123]
        print(f"Running grid search over coeff_lr_mult x weight_decay for seeds: {seeds_list}")
        run_grid_search(seeds_list, quick_epochs=args.epochs)
        print("Grid results saved; exiting as requested by --grid flag.")
        return


# ...existing code...

    # Degree sweep helper: test multiple polynomial degrees and save CSV
    def run_degree_sweep(degrees_list, seeds_list, quick_epochs=12, coeff_lr_mult=5.0, weight_decay=1e-6):
        rows = []
        for deg in degrees_list:
            print(f"\n=== Degree sweep: degree={deg} ===")
            for sd in seeds_list:
                print(f" Running seed={sd} for degree={deg}")
                torch.manual_seed(sd); np.random.seed(sd); random.seed(sd)

                # recreate data and loaders per-seed for full isolation
                dataset = MovingMNISTDataset(num_samples=args.num_samples, num_frames=20, image_size=32, num_digits=2)
                train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.4, random_state=sd)
                val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=sd)
                train_dataset = torch.utils.data.Subset(dataset, train_idx)
                val_dataset = torch.utils.data.Subset(dataset, val_idx)
                test_dataset = torch.utils.data.Subset(dataset, test_idx)

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

                # build models
                predrnn_model = PredRNNpp(in_channels=1, num_hidden=32, num_layers=2)
                stf_model = make_stfaglu_predrnn(in_channels=1, num_hidden=32, num_layers=2)
                # replace STFAGLU with desired degree
                stf_model.decoder = nn.Sequential(
                    SpatioTemporalFourierAdaPolyGLU(32, 32, spatial_kernel_size=3, max_1d_degree=deg, max_1d_freq=1),
                    nn.Conv2d(32, 1, 1), nn.Sigmoid()
                )

                t0 = time.time()
                train_model(predrnn_model, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"PredRNN_deg{deg}_seed{sd}", weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=args.amp)
                predrnn_time = time.time() - t0
                predrnn_test_loss = evaluate_model(predrnn_model, test_loader)
                predrnn_mse_ts, predrnn_psnr_ts, predrnn_ssim_ts, predrnn_lpips_ts = evaluate_model_metrics(predrnn_model, test_loader)

                t0 = time.time()
                train_model(stf_model, train_loader, val_loader, num_epochs=quick_epochs, lr=8e-4, model_name=f"STF_deg{deg}_seed{sd}", weight_decay=weight_decay, coeff_lr_mult=coeff_lr_mult, use_amp=args.amp)
                stf_time = time.time() - t0
                stf_test_loss = evaluate_model(stf_model, test_loader)
                stf_mse_ts, stf_psnr_ts, stf_ssim_ts, stf_lpips_ts = evaluate_model_metrics(stf_model, test_loader)

                # learned params
                gamma_poly = gamma_fourier = basis_mix = linear_mix = None
                for m in stf_model.modules():
                    if isinstance(m, SpatioTemporalFourierAdaPolyGLU):
                        try:
                            gamma_poly = float(m.gamma_poly.detach().cpu().item())
                            gamma_fourier = float(m.gamma_fourier.detach().cpu().item())
                            basis_mix = float(torch.sigmoid(m.basis_mix_logit).detach().cpu().item())
                            linear_mix = float(torch.sigmoid(m.linear_mix_logit).detach().cpu().item())
                        except Exception:
                            pass
                        break

                rows.append({
                    'degree': int(deg), 'seed': int(sd),
                    'predrnn_test_loss': float(predrnn_test_loss), 'stf_test_loss': float(stf_test_loss),
                    'predrnn_time': float(predrnn_time), 'stf_time': float(stf_time),
                    'gamma_poly': gamma_poly, 'gamma_fourier': gamma_fourier, 'basis_mix': basis_mix, 'linear_mix': linear_mix,
                    'predrnn_mse_ts': json.dumps(predrnn_mse_ts.tolist()), 'stf_mse_ts': json.dumps(stf_mse_ts.tolist()),
                    'predrnn_psnr_ts': json.dumps(predrnn_psnr_ts.tolist()), 'stf_psnr_ts': json.dumps(stf_psnr_ts.tolist())
                })

        # save CSV
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            script_dir = os.getcwd()
        out_path = os.path.join(script_dir, 'degree_sweep_results.csv')
        keys = rows[0].keys() if rows else []
        try:
            with open(out_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(keys))
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Degree sweep results saved to {out_path}")
        except Exception as e:
            print('Failed to write degree sweep CSV:', e)
        return rows

    # If user requested a degree sweep, run and exit early
    if args.degree_sweep:
        try:
            degrees_list = [int(x.strip()) for x in args.degree_sweep.split(',') if x.strip()]
        except Exception:
            degrees_list = [0, 1, 2, 3, 4]
        try:
            seeds_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except Exception:
            seeds_list = [42, 7, 123]
        print(f"Running degree sweep for degrees={degrees_list}, seeds={seeds_list}")
        run_degree_sweep(degrees_list, seeds_list, quick_epochs=12, coeff_lr_mult=args.coeff_lr_mult)
        print("Degree sweep complete. Exiting.")
        return

    # If user requested just multi-seed quick experiments, run and exit early
    if args.multi_seed:
        try:
            seeds_list = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except Exception:
            seeds_list = [42, 7, 123]
        print(f"Running multi-seed experiments for seeds: {seeds_list}")
        # Use the user-provided --epochs value for multi-seed runs so CLI controls epoch count
        run_multi_seed_experiments(seeds_list, quick_epochs=args.epochs, coeff_lr_mult=args.coeff_lr_mult)
        print("Multi-seed run complete. Exiting as requested by --multi-seed flag.")
        return

    print("\nCreating models...")
    predrnn_model = PredRNNpp(in_channels=1, num_hidden=32, num_layers=2)
    stfaglu_predrnn_model = STFAGLU_PredRNNpp(in_channels=1, num_hidden=32, num_layers=2)

    print(f"\nModel Parameters:")
    predrnn_params = sum(p.numel() for p in predrnn_model.parameters())
    stfaglu_predrnn_params = sum(p.numel() for p in stfaglu_predrnn_model.parameters())
    print(f"PredRNN++: {predrnn_params:,}")
    print(f"STFAGLU-PredRNN++: {stfaglu_predrnn_params:,}")
    print(f"Parameter Increase: {((stfaglu_predrnn_params - predrnn_params) / predrnn_params * 100):.1f}%")

    print("\n" + "=" * 40)
    print("TRAINING PRED-RNN++ (Baseline)")
    print("=" * 40)
    # Use the same learning rate for fair comparison; reduce STFAGLU weight decay slightly
    predrnn_train_losses, predrnn_val_losses, predrnn_time, predrnn_best_loss = train_model(
        predrnn_model, train_loader, val_loader, num_epochs=args.epochs, lr=8e-4, model_name="PredRNN++", weight_decay=1e-5, coeff_lr_mult=args.coeff_lr_mult)

    print("\n" + "=" * 40)
    print("TRAINING 3 STF VARIANTS (Original, Routed, Revolutionary)")
    print("=" * 40)
    # Train the canonical STFAGLU (used for final presentation) but also note
    # that multi-seed runs train all three variants for per-seed comparison.
    stfaglu_train_losses, stfaglu_val_losses, stfaglu_time, stfaglu_best_loss = train_model(
        stfaglu_predrnn_model, train_loader, val_loader, num_epochs=args.epochs, lr=8e-4, model_name="STFAGLU", weight_decay=1e-6, coeff_lr_mult=args.coeff_lr_mult)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    # Try to load best checkpoints (if saved during training) before final evaluation
    try:
        predrnn_ckpt = 'PredRNN++_best.pt'
        if os.path.exists(predrnn_ckpt):
            ck = torch.load(predrnn_ckpt, map_location=device)
            predrnn_model.load_state_dict(ck['model_state_dict'])
            print('Loaded best checkpoint for PredRNN++')
    except Exception:
        pass

    try:
        stfaglu_ckpt = 'STFAGLU_best.pt'
        if os.path.exists(stfaglu_ckpt):
            ck = torch.load(stfaglu_ckpt, map_location=device)
            stfaglu_predrnn_model.load_state_dict(ck['model_state_dict'])
            print('Loaded best checkpoint for STFAGLU')
    except Exception:
        pass

    predrnn_test_loss = evaluate_model(predrnn_model, test_loader)
    stfaglu_test_loss = evaluate_model(stfaglu_predrnn_model, test_loader)

    # Per-timestep metrics (MSE, PSNR, SSIM, LPIPS)
    predrnn_mse_ts, predrnn_psnr_ts, predrnn_ssim_ts, predrnn_lpips_ts = evaluate_model_metrics(predrnn_model, test_loader)
    stfaglu_mse_ts, stfaglu_psnr_ts, stfaglu_ssim_ts, stfaglu_lpips_ts = evaluate_model_metrics(stfaglu_predrnn_model, test_loader)

    print(f"\nFINAL COMPARISON:")
    print(f"PredRNN++ Test Loss:  {predrnn_test_loss:.6f}")
    print(f"STFAGLU Test Loss:   {stfaglu_test_loss:.6f}")

    print("\nPer-timestep MSE (first 5 timesteps):")
    print(f"PredRNN++ MSE: {predrnn_mse_ts[:5]}")
    print(f"STFAGLU MSE:  {stfaglu_mse_ts[:5]}")

    print("\nPer-timestep PSNR (dB, first 5 timesteps):")
    print(f"PredRNN++ PSNR: {predrnn_psnr_ts[:5]}")
    print(f"STFAGLU PSNR:  {stfaglu_psnr_ts[:5]}")

    improvement = ((predrnn_test_loss - stfaglu_test_loss) / predrnn_test_loss * 100)
    print(f"Test Loss Improvement: {improvement:.1f}%")
    print(f"Total Training Time (PredRNN++): {predrnn_time:.2f}s")
    print(f"Total Training Time (STFAGLU): {stfaglu_time:.2f}s")

    # Debug: print learned mixing/gain parameters from STFAGLU decoder
    try:
        stf_block = None
        for m in stfaglu_predrnn_model.modules():
            if isinstance(m, SpatioTemporalFourierAdaPolyGLU):
                stf_block = m
                break
        if stf_block is not None:
            print('\nSTFAGLU learned params:')
            print(f'  gamma_poly={stf_block.gamma_poly.item():.6f}, gamma_fourier={stf_block.gamma_fourier.item():.6f}')
            print(f'  basis_mix={torch.sigmoid(stf_block.basis_mix_logit).item():.6f}, linear_mix={torch.sigmoid(stf_block.linear_mix_logit).item():.6f}')
    except Exception:
        pass

    if improvement > 0:
        print("✅ STFAGLU-PredRNN++ performs BETTER than the PredRNN++ baseline.")
    else:
        print("❌ STFAGLU-PredRNN++ performs WORSE than the PredRNN++ baseline.")

    # Enhanced plotting for publication-ready visualization
    plt.figure(figsize=(18, 6))

    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot(predrnn_train_losses, 'b-', label='PredRNN++ Train', linewidth=2, alpha=0.8)
    plt.plot(predrnn_val_losses, 'b--', label='PredRNN++ Val', linewidth=2, alpha=0.8)
    plt.plot(stfaglu_train_losses, 'r-', label='STFAGLU Train', linewidth=2, alpha=0.8)
    plt.plot(stfaglu_val_losses, 'r--', label='STFAGLU Val', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Test performance
    plt.subplot(1, 3, 2)
    models = ['PredRNN++', 'STFAGLU']
    test_losses = [predrnn_test_loss, stfaglu_test_loss]
    colors = ['blue', 'red']

    bars = plt.bar(models, test_losses, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)
    plt.ylabel('Test Loss (MSE)')
    plt.title('Test Performance Comparison')

    for bar, loss in zip(bars, test_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                 f'{loss:.6f}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Parameter comparison
    plt.subplot(1, 3, 3)
    param_counts = [predrnn_params, stfaglu_predrnn_params]
    bars = plt.bar(models, param_counts, color=['lightblue', 'lightcoral'],
                   alpha=0.7, edgecolor='black', linewidth=1)
    plt.ylabel('Parameter Count')
    plt.title('Model Complexity')
    plt.yscale('log')

    for bar, count in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                 f'{count:,}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Sample predictions visualization
    print("\n" + "=" * 40)
    print("SAMPLE PREDICTIONS VISUALIZATION")
    print("=" * 40)

    def visualize_predictions(model, model_name, loader):
        model.eval()
        with torch.no_grad():
            inputs, targets = next(iter(loader))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, future_steps=targets.size(1))

            fig, axes = plt.subplots(2, 6, figsize=(15, 6))
            fig.suptitle(f'{model_name} Predictions', fontsize=16, fontweight='bold')

            sample_idx = 0
            time_steps = [0, 2, 4, 6, 8, 9]

            for i, t in enumerate(time_steps):
                axes[0, i].imshow(targets[sample_idx, t, 0].cpu().numpy(),
                                 cmap='viridis', vmin=0, vmax=1)
                axes[0, i].set_title(f'True t+{t+1}')
                axes[0, i].axis('off')

                axes[1, i].imshow(outputs[sample_idx, t, 0].cpu().numpy(),
                                 cmap='viridis', vmin=0, vmax=1)
                axes[1, i].set_title(f'Pred t+{t+1}')
                axes[1, i].axis('off')

            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(axes[0, 0].images[0], cax=cax)

            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.show()

    visualize_predictions(predrnn_model, "PredRNN++", test_loader)
    visualize_predictions(stfaglu_predrnn_model, "STFAGLU-PredRNN++", test_loader)

    # Run ablation experiments (keep original full model experiments intact)
    try:
        print('\n' + '=' * 40)
        print('RUNNING ABLATION EXPERIMENTS (quick)')
        print('=' * 40)
        # quick ablations for attribution; increase quick_epochs for stronger signal
        from functools import partial
        results = ablation_experiments(train_loader, val_loader, test_loader, quick_epochs=5)
        print('\nAblation summary:')
        for name, res in results.items():
            print(f"{name}: test_loss={res['test_loss']:.6f}, psnr_t1={res['psnr_ts'][0]:.2f} dB")
    except Exception as e:
        print('Ablation experiments failed to run:', e)

if __name__ == "__main__":
    main()


# --- Compatibility export: if a separate rev_v2 file exists, expose its class here
try:
    import importlib.util as _il, os as _os
    _rev_v2_path = _os.path.join(_os.path.dirname(__file__), 'spatio_temporal_activation_rev_v2.py')
    if _os.path.exists(_rev_v2_path):
        _spec_v2 = _il.spec_from_file_location('spatio_temporal_activation_rev_v2', _rev_v2_path)
        if _spec_v2 is not None:
            _mod_v2 = _il.module_from_spec(_spec_v2)
            _spec_v2.loader.exec_module(_mod_v2)
            if hasattr(_mod_v2, 'SpatioTemporalRevolutionaryGLU_v2'):
                SpatioTemporalRevolutionaryGLU_v2 = getattr(_mod_v2, 'SpatioTemporalRevolutionaryGLU_v2')
except Exception:
    # keep silent to avoid breaking users who import the module for other reasons
    pass
