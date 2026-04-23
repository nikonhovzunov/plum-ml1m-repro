"""Advanced multi-input RQ-VAE for PLUM-style Semantic IDs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RQVAEConfig:
    modality_dims: dict[str, int]
    latent_dim: int = 256
    branch_dim: int = 256
    branch_hidden_dims: tuple[int, ...] = (512, 512)
    fusion_hidden_dims: tuple[int, ...] = (512, 512)
    decoder_hidden_dims: tuple[int, ...] = (512, 512)
    codebook_sizes: tuple[int, ...] = (2048, 1024, 512, 256)
    dropout: float = 0.10
    use_description_mask: bool = True
    contrastive_dim: int = 128

    @property
    def n_levels(self) -> int:
        return len(self.codebook_sizes)


@dataclass
class RQVAEOutput:
    reconstructions: dict[str, torch.Tensor]
    h: torch.Tensor
    z_q: torch.Tensor
    z_q_st: torch.Tensor
    sids: torch.Tensor
    residuals: torch.Tensor
    quantized: torch.Tensor
    active_level_mask: torch.Tensor
    contrastive: torch.Tensor


def _make_mlp(
    in_dim: int,
    hidden_dims: tuple[int, ...],
    out_dim: int,
    *,
    dropout: float,
    final_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for hidden in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    if final_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class FeatureBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        branch_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.proj = _make_mlp(in_dim, hidden_dims[:1], branch_dim, dropout=dropout)
        block_hidden = hidden_dims[-1] if hidden_dims else branch_dim * 2
        self.block = ResidualMLPBlock(branch_dim, block_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class ResidualQuantizer(nn.Module):
    def __init__(self, latent_dim: int, codebook_sizes: tuple[int, ...]):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.codebooks = nn.ModuleList(
            [nn.Embedding(size, latent_dim) for size in self.codebook_sizes]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.codebooks:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    @property
    def n_levels(self) -> int:
        return len(self.codebooks)

    def forward(
        self,
        h: torch.Tensor,
        *,
        active_levels: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if active_levels is None:
            active_levels = self.n_levels
        active_levels = max(1, min(int(active_levels), self.n_levels))

        residual = h
        all_quantized: list[torch.Tensor] = []
        all_residuals: list[torch.Tensor] = []
        all_sids: list[torch.Tensor] = []

        for codebook in self.codebooks:
            weights = codebook.weight
            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * residual @ weights.t()
                + weights.pow(2).sum(dim=1).unsqueeze(0)
            )
            idx = distances.argmin(dim=1)
            q = codebook(idx)
            all_residuals.append(residual)
            all_quantized.append(q)
            all_sids.append(idx)
            residual = residual - q

        quantized = torch.stack(all_quantized, dim=1)
        residuals = torch.stack(all_residuals, dim=1)
        sids = torch.stack(all_sids, dim=1)
        active_mask = torch.zeros(self.n_levels, device=h.device, dtype=h.dtype)
        active_mask[:active_levels] = 1.0
        z_q = (quantized * active_mask.view(1, -1, 1)).sum(dim=1)
        return z_q, sids, residuals, quantized, active_mask


class AdvancedRQVAE(nn.Module):
    """Multi-modal RQ-VAE with branch encoders, fusion MLP, and contrastive head."""

    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.config = config
        self.modality_names = tuple(config.modality_dims.keys())
        self.branches = nn.ModuleDict(
            {
                name: FeatureBranch(
                    in_dim=dim,
                    branch_dim=config.branch_dim,
                    hidden_dims=config.branch_hidden_dims,
                    dropout=config.dropout,
                )
                for name, dim in config.modality_dims.items()
            }
        )

        fusion_in = config.branch_dim * len(self.modality_names)
        if config.use_description_mask:
            fusion_in += 1
        self.fusion = _make_mlp(
            fusion_in,
            config.fusion_hidden_dims,
            config.latent_dim,
            dropout=config.dropout,
        )
        self.quantizer = ResidualQuantizer(config.latent_dim, config.codebook_sizes)
        self.decoders = nn.ModuleDict(
            {
                name: _make_mlp(
                    config.latent_dim,
                    config.decoder_hidden_dims,
                    dim,
                    dropout=config.dropout,
                    final_norm=False,
                )
                for name, dim in config.modality_dims.items()
            }
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.contrastive_dim),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        description_mask: torch.Tensor | None = None,
        active_levels: int | None = None,
    ) -> RQVAEOutput:
        missing = set(self.modality_names) - set(inputs)
        if missing:
            raise ValueError(f"Missing modalities: {sorted(missing)}")

        branch_outputs = [self.branches[name](inputs[name]) for name in self.modality_names]
        fusion_parts = branch_outputs
        if self.config.use_description_mask:
            if description_mask is None:
                batch_size = branch_outputs[0].shape[0]
                description_mask = torch.ones(
                    batch_size,
                    1,
                    dtype=branch_outputs[0].dtype,
                    device=branch_outputs[0].device,
                )
            fusion_parts.append(description_mask.to(branch_outputs[0].dtype))

        h = self.fusion(torch.cat(fusion_parts, dim=1))
        z_q, sids, residuals, quantized, active_mask = self.quantizer(
            h,
            active_levels=active_levels,
        )
        z_q_st = h + (z_q - h).detach()
        reconstructions = {name: self.decoders[name](z_q_st) for name in self.modality_names}
        contrastive = F.normalize(self.contrastive_head(z_q_st), p=2, dim=1)
        return RQVAEOutput(
            reconstructions=reconstructions,
            h=h,
            z_q=z_q,
            z_q_st=z_q_st,
            sids=sids,
            residuals=residuals,
            quantized=quantized,
            active_level_mask=active_mask,
            contrastive=contrastive,
        )


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    recon: torch.Tensor
    rq: torch.Tensor
    codebook: torch.Tensor
    commit: torch.Tensor
    contrastive: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))


class AdvancedRQVAELoss(nn.Module):
    def __init__(
        self,
        *,
        beta: float = 0.25,
        codebook_weight: float = 1.0,
        recon_mse_weight: float = 0.25,
        recon_cosine_weight: float = 1.0,
        contrastive_weight: float = 0.05,
        contrastive_temperature: float = 0.07,
        modality_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.beta = beta
        self.codebook_weight = codebook_weight
        self.recon_mse_weight = recon_mse_weight
        self.recon_cosine_weight = recon_cosine_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.modality_weights = modality_weights or {}

    def reconstruction_loss(
        self,
        inputs: dict[str, torch.Tensor],
        output: RQVAEOutput,
    ) -> torch.Tensor:
        losses = []
        weights = []
        for name, x in inputs.items():
            x_hat = output.reconstructions[name]
            mse = F.mse_loss(x_hat, x)
            cosine = 1.0 - F.cosine_similarity(x_hat, x, dim=1).mean()
            weight = float(self.modality_weights.get(name, 1.0))
            losses.append(weight * (self.recon_mse_weight * mse + self.recon_cosine_weight * cosine))
            weights.append(weight)
        return torch.stack(losses).sum() / max(sum(weights), 1e-8)

    def rq_loss(self, output: RQVAEOutput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = output.active_level_mask.view(1, -1, 1)
        denom = active.sum().clamp_min(1.0)
        codebook = ((output.residuals.detach() - output.quantized).pow(2) * active).sum() / (
            output.residuals.shape[0] * denom * output.residuals.shape[2]
        )
        commit = self.beta * (
            (output.residuals - output.quantized.detach()).pow(2) * active
        ).sum() / (output.residuals.shape[0] * denom * output.residuals.shape[2])
        rq = self.codebook_weight * codebook + commit
        return rq, codebook, commit

    def bidirectional_info_nce(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchors.shape != positives.shape:
            raise ValueError("anchors and positives must have identical shapes.")
        batch_size = anchors.shape[0]
        logits = anchors @ positives.t()
        logits = logits / self.contrastive_temperature
        labels = torch.arange(batch_size, device=anchors.device)
        loss_a = F.cross_entropy(logits, labels, reduction="none")
        loss_p = F.cross_entropy(logits.t(), labels, reduction="none")
        losses = 0.5 * (loss_a + loss_p)
        if sample_weight is not None:
            weights = sample_weight.to(losses.device, losses.dtype).flatten()
            weights = weights / weights.mean().clamp_min(1e-8)
            losses = losses * weights
        return losses.mean()

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        output: RQVAEOutput,
        *,
        positive_inputs: dict[str, torch.Tensor] | None = None,
        positive_output: RQVAEOutput | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> LossBreakdown:
        recon = self.reconstruction_loss(inputs, output)
        rq, codebook, commit = self.rq_loss(output)
        total = recon + rq
        contrastive = torch.zeros((), device=output.h.device, dtype=output.h.dtype)

        if positive_inputs is not None and positive_output is not None:
            pos_recon = self.reconstruction_loss(positive_inputs, positive_output)
            pos_rq, pos_codebook, pos_commit = self.rq_loss(positive_output)
            recon = 0.5 * (recon + pos_recon)
            rq = 0.5 * (rq + pos_rq)
            codebook = 0.5 * (codebook + pos_codebook)
            commit = 0.5 * (commit + pos_commit)
            contrastive = self.bidirectional_info_nce(
                output.contrastive,
                positive_output.contrastive,
                sample_weight=sample_weight,
            )
            total = recon + rq + self.contrastive_weight * contrastive

        return LossBreakdown(
            total=total,
            recon=recon,
            rq=rq,
            codebook=codebook,
            commit=commit,
            contrastive=contrastive,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
