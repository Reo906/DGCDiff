from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evaluation.metrics import compute_metrics
from training.checkpoints import load_weights
from training.config import load_config
from training.core import (
    ForecastBatch,
    RunningMetrics,
    build_loader,
    build_model,
    choose_device,
    seed_all,
)

def evaluate(config: dict, checkpoint_path: str | None = None) -> dict:
    task = config["task"]
    settings = config["evaluation"]
    seed_all(int(settings.get("seed", config["experiment"]["seed"])))
    device = choose_device(config.get("runtime", {}).get("device", "auto"))
    model = build_model(config["model"], task, device)

    weights = checkpoint_path or settings["checkpoint"]
    load_weights(weights, model, device)
    model.eval()
    split = settings.get("split", "test")
    loader = build_loader(config["data"][split])

    context_length = int(task["context_length"])
    horizon = int(task["horizon"])
    sample_count = int(settings.get("num_samples", 20))

    output_directory = Path(settings.get("output_dir", "runs/evaluation")).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    interval = float(settings.get("interval", 0.9))
    aggregate = RunningMetrics()
    rows = []
    count = 0

    with torch.inference_mode():
        for tensors in tqdm(loader, desc="Evaluate", leave=False):
            batch = ForecastBatch.from_dataset(tensors, device)
            draws = [model.predict(batch, context_length, horizon) for _ in range(sample_count)]
            predictions = torch.stack([draw.samples for draw in draws], dim=1)
            targets = batch.x[:, context_length : context_length + horizon]

            for example in range(targets.shape[0]):
                samples, truth = predictions[example], targets[example]
                mean, std = batch.x_stats[example, 0]
                samples = samples * std + mean
                truth = truth * std + mean

                scores = compute_metrics(
                    samples.cpu(),
                    truth.cpu(),
                    interval=interval,
                )
                aggregate.add(scores)
                rows.append({"window_index": count, **scores})
                count += 1

    column_names = list(dict.fromkeys(key for row in rows for key in row))
    with (output_directory / "per_window.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "n_windows": count,
        "metrics": aggregate.result(),
    }
    (output_directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    evaluate(load_config(args.config), checkpoint_path=args.checkpoint)
