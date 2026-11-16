from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class SampleOutput:
    samples: torch.Tensor


class ForecastContractMixin:
    @torch.no_grad()
    def predict(self, batch, context_length: int, horizon: int) -> SampleOutput:
        context = batch.x[:, :context_length]
        masked_x = torch.cat(
            (context, context.new_zeros(context.shape[0], horizon, *context.shape[2:])),
            dim=1,
        )
        result = self.sample(
            x=masked_x,
            adjacency=batch.adjacency,
            tod=batch.tod[:, :context_length + horizon],
            dow=batch.dow[:, :context_length + horizon],
            context_len=context_length,
            pred_length=horizon,
        )
        return result
