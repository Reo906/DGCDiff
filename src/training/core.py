from __future__ import annotations

import importlib
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class ForecastBatch:
    adjacency: torch.Tensor
    x: torch.Tensor
    tod: torch.Tensor
    dow: torch.Tensor
    x_stats: torch.Tensor | None = None

    @classmethod
    def from_dataset(cls, tensors, device: torch.device | str) -> "ForecastBatch":
        return cls(
            *(tensor.to(device=device, dtype=torch.float32, non_blocking=True) for tensor in tensors)
        )


class RunningMetrics:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, values: dict, weight: int = 1) -> None:
        for name, value in values.items():
            scalar = float(torch.as_tensor(value).detach().mean().cpu())
            self.sums[name] = self.sums.get(name, 0.0) + scalar * weight
            self.counts[name] = self.counts.get(name, 0) + weight

    def result(self) -> dict[str, float]:
        return {name: total / self.counts[name] for name, total in self.sums.items()}


def construct(spec: dict, **injected):
    path = spec["target"]
    module_name, class_name = path.rsplit(".", 1)
    constructor = getattr(importlib.import_module(module_name), class_name)
    return constructor(**{**(spec.get("parameters") or {}), **injected})


def build_model(spec: dict, task: dict, device):
    parameters = dict(spec.get("parameters") or {})
    parameters["context_len"] = int(task["context_length"])
    return construct({"target": spec["target"], "parameters": parameters}).to(device)


def build_loader(spec: dict):
    dataset = construct(spec["dataset"])
    return DataLoader(dataset, **dict(spec.get("loader") or {}))


def choose_device(requested="auto") -> torch.device:
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
