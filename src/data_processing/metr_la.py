from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class METRLADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        adjacency_path: str,
        context_length: int,
        horizon: int,
        start_index: int,
        end_index: int,
        temporal_stride: int = 1,
        norm_mean: float = 54.4059,
        norm_std: float = 19.4943,
    ):
        self.window_length = int(context_length) + int(horizon)
        self.temporal_stride = int(temporal_stride)
        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)

        frame = pd.read_hdf(Path(data_path))
        start, end = int(start_index), int(end_index)
        frame = frame.iloc[start:end]

        values = frame.apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
        self.values = values.to_numpy(dtype=np.float32, copy=True)
        self.timestamps = pd.DatetimeIndex(values.index)
        self.adjacency = self._load_adjacency(
            Path(adjacency_path), [str(value) for value in values.columns]
        )
        self.starts = np.arange(
            0, len(self.values) - self.window_length + 1, self.temporal_stride
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[int(index)])
        stop = start + self.window_length
        raw = self.values[start:stop]
        x = ((raw - self.norm_mean) / self.norm_std)[..., None]

        timestamps = self.timestamps[start:stop]
        seconds = (timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second).to_numpy(
            dtype=np.float32
        )
        tod = seconds / 86400.0
        dow = timestamps.dayofweek.to_numpy(dtype=np.float32) / 7.0

        sample = (
            torch.from_numpy(self.adjacency),
            torch.from_numpy(x),
            torch.from_numpy(tod),
            torch.from_numpy(dow),
        )
        stats = torch.tensor([[self.norm_mean, self.norm_std]], dtype=torch.float32)
        return (*sample, stats)

    @staticmethod
    def _load_adjacency(path: Path, columns: list[str]) -> np.ndarray:
        with path.open("rb") as file:
            _, sensor_id_to_index, matrix = pickle.load(file, encoding="latin1")
        index = {str(key): int(value) for key, value in sensor_id_to_index.items()}
        order = np.asarray([index[node_id] for node_id in columns], dtype=np.int64)
        adjacency = np.asarray(matrix, dtype=np.float32)[np.ix_(order, order)]
        adjacency = np.clip(adjacency, 0.0, None)
        row_sum = adjacency.sum(axis=1, keepdims=True)
        return (adjacency / np.maximum(row_sum, 1e-6)).astype(np.float32)
