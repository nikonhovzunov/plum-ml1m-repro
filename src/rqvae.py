import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_d: int, h_d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 256),
            nn.ReLU(),
            nn.Linear(256, h_d),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_d: int, h_d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_d, 256),
            nn.ReLU(),
            nn.Linear(256, out_d),
        )

    def forward(self, h):
        return self.net(h)


class MultiResolutionCodebooks(nn.Module):
    def __init__(self, h_d: int, n_levels: int):
        super().__init__()
        self.h_d = h_d
        self.Progressive_Masking = False
        self.epoch_devider = 100
        self.current_epoch = 0
        self.n_levels = n_levels
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(2048 // (2**level), self.h_d)
                for level in range(self.n_levels)
            ]
        )

    def set_progressive_masking(self, on: bool, current_epoch, epoch_devider: int):
        self.current_epoch = current_epoch
        self.epoch_devider = epoch_devider
        self.Progressive_Masking = on

    def forward(self, h):
        SIDs = [0] * self.n_levels
        r_list = [0] * self.n_levels
        q_list = [0] * self.n_levels
        z_q_sum_arr = [0] * self.n_levels

        for i in range(self.n_levels):
            r_in = h
            current_codebook = self.codebooks[i].weight
            idxs = torch.argmin(
                torch.sum(
                    ((current_codebook[None, :, :] - r_in[:, None, :]) ** 2),
                    dim=2,
                ),
                dim=1,
            )
            q = current_codebook[idxs]

            h = r_in - q
            z_q_sum_arr[i] = q

            SIDs[i] = idxs
            r_list[i] = r_in
            q_list[i] = q

        z_q_sum_arr = torch.stack(z_q_sum_arr, dim=1)

        if self.Progressive_Masking:
            r_max = min(self.n_levels, self.current_epoch // self.epoch_devider + 1)
            r = torch.randint(1, r_max + 1, (1,)).item()
            z_q = z_q_sum_arr[:, :r, :].sum(dim=1)
        else:
            z_q = z_q_sum_arr.sum(dim=1)

        SIDs = torch.stack(SIDs, dim=1)
        r_list = torch.stack(r_list, dim=1)
        q_list = torch.stack(q_list, dim=1)

        return z_q, SIDs, r_list, q_list


class RQVAE(nn.Module):
    def __init__(self, in_d: int, h_d: int, n_levels: int):
        super().__init__()
        self.encoder = Encoder(in_d, h_d)
        self.codebooks = MultiResolutionCodebooks(h_d, n_levels)
        self.decoder = Decoder(in_d, h_d)

    def forward(self, x):
        h = self.encoder(x)
        z_q, SIDs, r_list, q_list = self.codebooks(h)

        # STE: forward uses z_q, backward uses h.
        z_q_st = h + (z_q - h).detach()

        x_hat = self.decoder(z_q_st)

        return x_hat, h, z_q, SIDs, r_list, q_list

    def set_progressive_masking(self, on: bool, current_epoch, epoch_devider: int):
        self.codebooks.set_progressive_masking(on, current_epoch, epoch_devider)


class RQVAELoss(nn.Module):
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta

    def co_occurence_contrastive_regularization(self, p, p_pos):
        B = p.size(0)
        z = torch.cat([p, p_pos], dim=0)

        logits = z @ z.t()
        logits.fill_diagonal_(-1e9)

        logits = logits - logits.max(dim=1, keepdim=True).values

        pos = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
        num = torch.exp(logits[torch.arange(2 * B, device=z.device), pos])
        denom = torch.exp(logits).sum(dim=1)
        loss = -(num / denom).mean()
        return loss

    def forward(self, x, x_hat, r_list, q_list, p=None, p_pos=None):
        L = r_list.shape[1]
        recon_loss = F.mse_loss(x, x_hat)
        codebook_loss = F.mse_loss(r_list.detach(), q_list) * L
        commit_loss = self.beta * F.mse_loss(r_list, q_list.detach()) * L
        rq_loss = codebook_loss + commit_loss
        total = recon_loss + rq_loss

        if (p is not None) and (p_pos is not None):
            con_loss = self.co_occurence_contrastive_regularization(p, p_pos)
        else:
            con_loss = torch.tensor(0.0, device=x.device)

        return total, recon_loss, codebook_loss, commit_loss, con_loss
