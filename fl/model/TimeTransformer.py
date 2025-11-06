"""Anomaly Transformer model definitions.

Copied and lightly adapted from the original Anomaly-Transformer project:
https://github.com/huyln15/Anomaly-Transformer
"""

from __future__ import annotations

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TriangularCausalMask:
    def __init__(self, batch: int, length: int, device: str = "cpu") -> None:
        mask_shape = [batch, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(
        self,
        win_size: int,
        mask_flag: bool = True,
        scale: float = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size

        distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                distances[i][j] = abs(i - j)
        self.register_buffer("distances", distances)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        sigma: torch.Tensor,
        attn_mask: TriangularCausalMask,
    ):
        bsz, length, heads, embed = queries.shape
        _, steps, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(embed)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(bsz, length, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H -> B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.to(queries.device).unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior**2 / 2 / (sigma**2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        series = torch.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        series = series / (series.sum(dim=-1, keepdim=True) + 1e-12)
        v = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (v.contiguous(), series, prior, sigma)
        return (v.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
    ) -> None:
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: TriangularCausalMask):
        bsz, length, _ = queries.shape
        _, steps, _ = keys.shape
        heads = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(bsz, length, heads, -1)
        keys = self.key_projection(keys).view(bsz, steps, heads, -1)
        values = self.value_projection(values).view(bsz, steps, heads, -1)
        sigma = self.sigma_projection(x).view(bsz, length, heads)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask,
        )
        out = out.view(bsz, length, -1)

        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: TriangularCausalMask = None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers: nn.ModuleList, norm_layer: nn.LayerNorm = None) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: TriangularCausalMask = None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(
        self,
        win_size: int,
        enc_in: int,
        c_out: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = True,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.output_attention = output_attention

        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x: torch.Tensor):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        return enc_out
