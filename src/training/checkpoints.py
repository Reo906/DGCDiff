from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save(path, model, optimizer, scheduler, epoch: int, best_value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_value": best_value,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, path)


def load_weights(path, model, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return state


def restore(path, model, optimizer, scheduler, device) -> tuple[int, float | None]:
    state = load_weights(path, model, device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state["epoch"] + 1, state["best_value"]
