import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from training.config import load_config
from training.engine import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
