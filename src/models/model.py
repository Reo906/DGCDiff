import math

import torch
from torch import nn

from models.networks import GraphAwareDenoiseNet, SpatialBlock, NodeWiseLSTMBlock, DynamicGraphEvolution
from models.contract import ForecastContractMixin, LossOutput, SampleOutput
from models.gaussian_diffusion import GaussianDiffusion


class InputEncoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, u_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, u_dim),
        )

    def forward(self, values):
        return self.net(values)


class TemporalFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim, u_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, u_dim),
        )

    def forward(self, time_of_day, day_of_week):
        return self.net(torch.stack((time_of_day, day_of_week), dim=-1))


class DGCDiff(ForecastContractMixin, nn.Module):
    def __init__(
        self,
        x_dim,
        h_dim=128,
        u_dim=64,
        diff_dim=None,
        hidden_dim=128,
        diff_steps=50,
        dropout=0.1,
        context_len=12,
        horizon_decay=0.98,
        graph_state_dim=None,
        graph_attn_dim=32,
        graph_update_scale=1.0,
        num_graph_blocks=2,
        time_hidden_dim=32,
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.h_dim = int(h_dim)
        self.horizon_decay = float(horizon_decay)
        state_dim = 2 * self.h_dim
        graph_state_dim = int(graph_state_dim or h_dim)

        self.input_encoder = InputEncoder(x_dim, hidden_dim, u_dim, dropout)
        self.time_encoder = TemporalFeatureEncoder(time_hidden_dim, u_dim, dropout)
        self.spatial_block = SpatialBlock(u_dim, h_dim, h_dim, h_dim, dropout)
        self.intra_lstm = NodeWiseLSTMBlock(u_dim, h_dim, dropout)
        self.inter_lstm = NodeWiseLSTMBlock(h_dim, h_dim, dropout)
        self.dynamic_graph_evolution = DynamicGraphEvolution(
            h_dim,
            u_dim,
            hidden_dim,
            graph_state_dim,
            graph_attn_dim,
            dropout,
            graph_update_scale,
        )

        self.prior_mean = nn.Linear(state_dim, h_dim)
        self.prior_logvar = nn.Linear(state_dim, h_dim)
        self.posterior_x_encoder = nn.Linear(x_dim, h_dim)
        self.enc_mean = nn.Linear(state_dim + h_dim, h_dim)
        self.enc_logvar = nn.Linear(state_dim + h_dim, h_dim)
        self.z_proj = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Softplus())

        self.denoise_net = GraphAwareDenoiseNet(
            x_dim,
            h_dim,
            h_dim,
            diff_step_emb_dim=diff_dim or h_dim,
            dropout=dropout,
            num_graph_blocks=num_graph_blocks,
        )
        self.diffusion = GaussianDiffusion(self.denoise_net, diff_steps=diff_steps)

    def _initial_state(self, batch_size, nodes, reference):
        intra = self.intra_lstm.init_states(batch_size, nodes, reference.device, reference.dtype)
        inter = self.inter_lstm.init_states(batch_size, nodes, reference.device, reference.dtype)
        graph = reference.new_zeros(
            batch_size, nodes, self.dynamic_graph_evolution.graph_state_dim
        )
        return *intra, *inter, graph

    def _encode(self, values, time_features):
        return self.input_encoder(values) + time_features.unsqueeze(1)

    def _step(self, h_intra, c_intra, h_inter, c_inter, adjacency, graph_state, inputs):
        h_intra, c_intra = self.intra_lstm(inputs, h_intra, c_intra)
        spatial = self.spatial_block(inputs, h_intra, adjacency)
        h_inter, c_inter = self.inter_lstm(spatial, h_inter, c_inter)
        adjacency, graph_state = self.dynamic_graph_evolution(
            h_inter, inputs, adjacency, graph_state
        )
        return h_intra, c_intra, h_inter, c_inter, adjacency, graph_state

    @staticmethod
    def _stabilize_logvar(logvar):
        return torch.nan_to_num(logvar, nan=0, posinf=10, neginf=-10).clamp(-10, 10)

    def _prior(self, state):
        return self.prior_mean(state), self._stabilize_logvar(self.prior_logvar(state))

    def _posterior(self, state, target):
        encoded_target = self.posterior_x_encoder(target)
        combined = torch.cat((state, encoded_target), dim=-1)
        return self.enc_mean(combined), self._stabilize_logvar(self.enc_logvar(combined))

    def _sample_latent(self, mean, logvar):
        std = (0.5 * logvar).exp().clamp_min(1e-6)
        latent = mean + torch.randn_like(std) * std
        return self.z_proj(latent)

    @staticmethod
    def _kld(mean_1, logvar_1, mean_2, logvar_2):
        kld = (logvar_2 - logvar_1) / 2
        kld += (logvar_1.exp() + (mean_1 - mean_2).square()) / (2 * logvar_2.exp()) - 0.5
        return kld.mean()

    def forward(
        self,
        x,
        adjacency,
        tod,
        dow,
        kld_weight=0.0,
        rec_weight=1.0,
    ):
        batch_size, total_steps, nodes, _ = x.shape
        time_features = self.time_encoder(tod, dow)
        h_intra, c_intra, h_inter, c_inter, graph_state = self._initial_state(
            batch_size, nodes, x
        )

        kld = x.new_tensor(0.0)
        pred_loss = x.new_tensor(0.0)
        weight_sum = x.new_tensor(0.0)

        for step in range(self.context_len):
            previous_adjacency = adjacency
            inputs = self._encode(x[:, step], time_features[:, step])
            h_intra, c_intra, h_inter, c_inter, adjacency, graph_state = self._step(
                h_intra,
                c_intra,
                h_inter,
                c_inter,
                previous_adjacency,
                graph_state,
                inputs,
            )

        for step in range(self.context_len, total_steps):
            previous = h_intra, c_intra, h_inter, c_inter
            previous_adjacency = adjacency
            state = torch.cat((h_intra, h_inter), dim=-1)
            prior_mean, prior_logvar = self._prior(state)
            posterior_mean, posterior_logvar = self._posterior(state, x[:, step])

            kld_step = self._kld(
                posterior_mean, posterior_logvar, prior_mean, prior_logvar
            )
            kld += kld_step
            posterior_condition = self._sample_latent(posterior_mean, posterior_logvar)
            prior_condition = self._sample_latent(prior_mean, prior_logvar)
            posterior_loss = self.diffusion.log_prob(
                x[:, step], posterior_condition, previous_adjacency
            )
            prior_loss = self.diffusion.log_prob(
                x[:, step], prior_condition, previous_adjacency
            )

            weight = self.horizon_decay ** (step - self.context_len)
            pred_loss += weight * (0.2 * posterior_loss + 0.8 * prior_loss)
            weight_sum += weight
            if step == total_steps - 1:
                continue

            inputs = self._encode(x[:, step], time_features[:, step])
            h_intra, c_intra, h_inter, c_inter, adjacency, graph_state = self._step(
                *previous, previous_adjacency, graph_state, inputs
            )

        forecast_steps = total_steps - self.context_len
        decoder_loss = pred_loss / weight_sum.clamp_min(1e-6)
        kld_loss = kld / max(forecast_steps, 1)
        loss = rec_weight * decoder_loss + kld_weight * kld_loss
        return LossOutput(
            loss,
            {
                "loss": loss.detach(),
                "kl": kld_loss.detach(),
                "decoder": decoder_loss.detach(),
            },
        )

    @torch.no_grad()
    def sample(
        self,
        x,
        adjacency,
        context_len,
        pred_length,
        tod,
        dow,
    ):
        batch_size, _, nodes, _ = x.shape
        time_features = self.time_encoder(tod, dow)
        h_intra, c_intra, h_inter, c_inter, graph_state = self._initial_state(
            batch_size, nodes, x
        )

        for step in range(context_len):
            inputs = self._encode(x[:, step], time_features[:, step])
            h_intra, c_intra, h_inter, c_inter, adjacency, graph_state = self._step(
                h_intra, c_intra, h_inter, c_inter, adjacency, graph_state, inputs
            )

        predictions = []
        reference = x[:, context_len - 1]
        for step in range(pred_length):
            previous = h_intra, c_intra, h_inter, c_inter
            previous_adjacency = adjacency
            prior_mean, prior_logvar = self._prior(torch.cat((h_intra, h_inter), dim=-1))
            condition = self._sample_latent(prior_mean, prior_logvar)
            prediction = self.diffusion.sample(reference.shape, condition, previous_adjacency)
            predictions.append(prediction)

            inputs = self._encode(prediction, time_features[:, context_len + step])
            h_intra, c_intra, h_inter, c_inter, adjacency, graph_state = self._step(
                *previous, previous_adjacency, graph_state, inputs
            )
            reference = prediction

        return SampleOutput(torch.stack(predictions, dim=1))
