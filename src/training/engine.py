from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from . import checkpoints
from .config import save_config
from .core import ForecastBatch, RunningMetrics, build_loader, build_model, choose_device, construct, seed_all


def run(config: dict) -> Path:
    seed_all(int(config["experiment"]["seed"]))
    device = choose_device(config.get("runtime", {}).get("device", "auto"))
    model = build_model(config["model"], config["task"], device)
    objective = dict(config.get("objective", {}).get("parameters") or {})
    train_loader = build_loader(config["data"]["train"])
    val_loader = build_loader(config["data"]["validation"])

    settings = config["checkpoint"]
    output_dir = Path(settings.get("directory", "runs/default")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    resume = settings.get("resume_from")
    optimizer = construct(config["optimizer"], params=model.parameters())
    scheduler_spec = config["scheduler"]
    scheduler = construct(scheduler_spec, optimizer=optimizer)
    start_epoch, best = (
        checkpoints.restore(resume, model, optimizer, scheduler, device)
        if resume
        else (0, float("inf"))
    )

    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        results = {}
        for split, loader in (("train", train_loader), ("val", val_loader)):
            training = split == "train"
            model.train(training)
            metrics = RunningMetrics()
            for tensors in tqdm(loader, desc=f"{split} {epoch + 1}", leave=False):
                batch = ForecastBatch.from_dataset(tensors, device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    output = model(
                        x=batch.x, adjacency=batch.adjacency, tod=batch.tod,
                        dow=batch.dow, **objective,
                    )
                    loss = output.loss.mean()
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(config["training"]["gradient_clipping"])
                        )
                        optimizer.step()
                metrics.add(output.metrics, weight=batch.x.shape[0])
            results.update({f"{split}/{key}": value for key, value in metrics.result().items()})

        current = results["val/loss"]
        improved = current < best
        if improved:
            best = current
        scheduler.step(current)
        summary = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], **results}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
        checkpoints.save(output_dir / "latest.pth", model, optimizer, scheduler, epoch, best)
        if improved:
            checkpoints.save(output_dir / "best.pth", model, optimizer, scheduler, epoch, best)
    return output_dir
