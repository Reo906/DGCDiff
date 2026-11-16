import torch


def compute_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    *,
    interval: float = 0.9,
) -> dict[str, float]:
    valid = torch.isfinite(truth) & (truth != 0)
    error = pred.mean(dim=0)[valid] - truth[valid]
    observation = (pred - truth.unsqueeze(0)).abs().mean(dim=0)
    pairwise = (pred.unsqueeze(0) - pred.unsqueeze(1)).abs().mean(dim=(0, 1))
    tail = (1 - interval) / 2
    lower, upper = torch.quantile(pred, pred.new_tensor([tail, 1 - tail]), dim=0)
    percentage = round(interval * 100)
    return {
        "MAE": error.abs().mean().item(),
        "RMSE": error.square().mean().sqrt().item(),
        "MAPE": (error.abs() / truth[valid].abs().clamp_min(1e-8)).mean().mul(100).item(),
        "CRPS": (observation - 0.5 * pairwise)[valid].mean().item(),
        f"PICP_{percentage}": ((truth >= lower) & (truth <= upper))[valid].float().mean().item(),
        f"MPIW_{percentage}": (upper - lower)[valid].mean().item(),
    }
