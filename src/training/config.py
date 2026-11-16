from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for split in ("train", "validation", "test"):
        parameters = config["data"][split]["dataset"]["parameters"]
        for key in ("data_path", "adjacency_path"):
            parameters[key] = str((path.parent / parameters[key]).resolve())
    return config


def save_config(config: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
