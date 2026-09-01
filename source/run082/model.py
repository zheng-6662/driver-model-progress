from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


COMMAND_INDEX = (0, 1)
CONTEXT_INDEX = (2, 5, 6, 7)
RESPONSE_INDEX = (3, 4)


def prepare_prefix(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """事件内稳健归一化；绝对尺度保留为22维向量。"""
    values = np.asarray(sequence[:, :7], dtype=np.float32)
    valid = np.isfinite(values)
    masked = np.ma.masked_invalid(values)
    median = np.ma.median(masked, axis=2).filled(0.0).astype(np.float32)
    centered = np.ma.masked_invalid(np.abs(values - median[:, :, None]))
    mad = (1.4826 * np.ma.median(centered, axis=2).filled(0.0)).astype(np.float32)
    mad = np.where(mad > 1e-6, mad, 1.0).astype(np.float32)
    filled = np.where(valid, values, median[:, :, None])
    normalized = (filled - median[:, :, None]) / mad[:, :, None]
    normalized = np.clip(normalized, -12.0, 12.0).astype(np.float32)
    road_mask = np.asarray(sequence[:, 7:8], dtype=np.float32)
    normalized_sequence = np.concatenate([normalized, road_mask], axis=1)
    release_value = filled[:, :, -1].astype(np.float32)
    absolute = np.column_stack([median, mad, release_value, road_mask[:, 0, -1]]).astype(np.float32)
    response_valid = valid[:, list(RESPONSE_INDEX)].astype(np.float32)
    return normalized_sequence, absolute, response_valid


def fit_absolute_scaler(values: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values[fit_indices], axis=0)
    scale = 1.4826 * np.median(np.abs(values[fit_indices] - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def transform_absolute(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - center) / scale, -12.0, 12.0).astype(np.float32)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class CausalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class RoleStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        kernel_size: int,
        dilations: tuple[int, int],
        dropout: float,
    ):
        super().__init__()
        self.first = CausalConv(in_channels, width, kernel_size, dilations[0])
        self.first_norm = ChannelLayerNorm(width)
        self.second = CausalConv(width, width, kernel_size, dilations[1])
        self.second_norm = ChannelLayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(F.silu(self.first_norm(self.first(values))))
        update = self.dropout(F.silu(self.second_norm(self.second(hidden))))
        return hidden + update


def pool_role(hidden: torch.Tensor) -> torch.Tensor:
    return torch.cat([hidden[:, :, -1], hidden.mean(dim=2), hidden[:, :, -5:].mean(dim=2)], dim=1)


class RoleTCN(nn.Module):
    def __init__(self, config: dict[str, object], absolute_dim: int = 22):
        super().__init__()
        width = int(config["role_width"])
        kernel = int(config["kernel_size"])
        dilations = tuple(int(value) for value in config["dilations"])
        dropout = float(config["dropout"])
        hidden_width = int(config["hidden_width"])
        self.command = RoleStem(2, width, kernel, dilations, dropout)
        self.context = RoleStem(4, width, kernel, dilations, dropout)
        self.response = RoleStem(2, width, kernel, dilations, dropout)
        input_dim = width * 3 * 3 + absolute_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_width, 20)

    def forward(
        self,
        sequence: torch.Tensor,
        absolute: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        command = self.command(sequence[:, list(COMMAND_INDEX)])
        context = self.context(sequence[:, list(CONTEXT_INDEX)])
        response = self.response(sequence[:, list(RESPONSE_INDEX)])
        state = torch.cat([pool_role(command), pool_role(context), pool_role(response), absolute], dim=1)
        return {"prediction": self.head(self.fusion(state))}


class PlainRawTCN(nn.Module):
    def __init__(self, config: dict[str, object], absolute_dim: int = 22):
        super().__init__()
        kernel = int(config["kernel_size"])
        dilations = tuple(int(value) for value in config["dilations"])
        dropout = float(config["dropout"])
        hidden_width = int(config["hidden_width"])
        self.stem = RoleStem(8, 32, kernel, dilations, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(32 * 3 + absolute_dim, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_width, 20)

    def forward(
        self,
        sequence: torch.Tensor,
        absolute: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = torch.cat([pool_role(self.stem(sequence)), absolute], dim=1)
        return {"prediction": self.head(self.fusion(state))}


class LGRS(nn.Module):
    def __init__(self, config: dict[str, object], lag_samples: list[int], absolute_dim: int = 22):
        super().__init__()
        width = int(config["role_width"])
        residual_width = int(config["residual_width"])
        hidden_width = int(config["hidden_width"])
        kernel = int(config["kernel_size"])
        dilations = tuple(int(value) for value in config["dilations"])
        dropout = float(config["dropout"])
        rank = int(config["relation_rank"])
        self.lag_samples = tuple(int(value) for value in lag_samples)
        self.command = RoleStem(2, width, kernel, dilations, dropout)
        self.context = RoleStem(4, width, kernel, dilations, dropout)
        self.residual = RoleStem(2, residual_width, kernel, dilations, dropout)
        context_token_dim = width * 2 + absolute_dim
        self.context_token = nn.Linear(context_token_dim, 8)
        self.response_factor_raw = nn.Parameter(torch.zeros(2, rank))
        self.gain_head = nn.Linear(8, 2 * len(self.lag_samples) * rank)
        self.bias_head = nn.Linear(8, 2)
        relation_pool_dim = residual_width * 3 + 6
        self.relation_adapter = nn.Sequential(nn.Linear(relation_pool_dim, 40), nn.SiLU())
        fusion_dim = width * 3 * 2 + 40 + absolute_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_width, 20)

    @staticmethod
    def lagged(values: torch.Tensor, lag: int) -> torch.Tensor:
        if lag == 0:
            return values
        prefix = values[:, :, :1].expand(-1, -1, lag)
        return torch.cat([prefix, values[:, :, :-lag]], dim=2)

    def relation_operator(
        self,
        command_values: torch.Tensor,
        context_hidden: torch.Tensor,
        absolute: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_input = torch.cat(
            [context_hidden[:, :, -1], context_hidden.mean(dim=2), absolute], dim=1
        )
        token = F.silu(self.context_token(context_input))
        rank = self.response_factor_raw.shape[1]
        conditional = 0.5 * torch.tanh(self.gain_head(token)).view(
            -1, 2, len(self.lag_samples), rank
        )
        response_factor = torch.tanh(self.response_factor_raw)
        gains = torch.einsum("jr,bklr->bjkl", response_factor, conditional)
        lagged = torch.stack(
            [self.lagged(command_values, lag) for lag in self.lag_samples], dim=2
        )
        expected = self.bias_head(token)[:, :, None] + torch.einsum("bjkl,bklt->bjt", gains, lagged)
        return expected, gains, token

    def forward(
        self,
        sequence: torch.Tensor,
        absolute: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        command_values = sequence[:, list(COMMAND_INDEX)]
        response_values = sequence[:, list(RESPONSE_INDEX)]
        command_hidden = self.command(command_values)
        context_hidden = self.context(sequence[:, list(CONTEXT_INDEX)])
        expected, gains, _ = self.relation_operator(command_values, context_hidden, absolute)
        residual_values = (response_values - expected) * response_mask
        residual_hidden = self.residual(residual_values)

        absolute_gain = gains.abs()
        denominator = absolute_gain.sum(dim=(2, 3)).clamp_min(1e-6)
        lag_ms = torch.tensor(
            [lag * 20.0 for lag in self.lag_samples], device=gains.device, dtype=gains.dtype
        )
        centroid = (
            absolute_gain.sum(dim=2) * lag_ms[None, None, :]
        ).sum(dim=2) / denominator / 240.0
        total = torch.log1p(denominator)
        balance = gains.sum(dim=(2, 3)) / denominator
        gain_summary = torch.stack([centroid, total, balance], dim=2).reshape(gains.shape[0], 6)
        relation_state = self.relation_adapter(
            torch.cat([pool_role(residual_hidden), gain_summary], dim=1)
        )
        state = torch.cat(
            [pool_role(command_hidden), pool_role(context_hidden), relation_state, absolute], dim=1
        )
        prediction = self.head(self.fusion(state))
        return {
            "prediction": prediction,
            "expected_response": expected,
            "normalized_response": response_values,
            "response_mask": response_mask,
            "gains": gains,
        }


def curve_loss(prediction: torch.Tensor, truth: torch.Tensor, difference_weight: float) -> torch.Tensor:
    point = torch.mean(torch.abs(prediction - truth))
    difference = F.huber_loss(torch.diff(prediction, dim=1), torch.diff(truth, dim=1))
    return point + difference_weight * difference


def relation_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    error = F.huber_loss(
        output["expected_response"],
        output["normalized_response"],
        reduction="none",
    )
    mask = output["response_mask"]
    return torch.sum(error * mask) / mask.sum().clamp_min(1.0)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
