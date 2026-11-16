# Dynamic Graph-Conditioned Diffusion (DGCDiff)

PyTorch code for training and evaluating DGCDiff on the public METR-LA traffic
forecasting benchmark. 

## Setup

```bash
python -m pip install -r requirements.txt
```

Download the standard METR-LA files from the
[DCRNN repository](https://github.com/liyaguang/DCRNN) and place them at:

```text
data/METR_LA/METR-LA.h5
data/METR_LA/adj_METR-LA.pkl
```

## Train

```bash
python train.py --config configs/train_metr_la.yaml
```

The example configuration uses the chronological 70/10/20 METR-LA split,
12 observed steps, 12 forecast steps, and normalization statistics calculated
from the training split. Checkpoints and the configuration are written
under `runs/metr_la_dgcdiff/`.

## Evaluate

```bash
python evaluate.py \
  --config configs/train_metr_la.yaml \
  --checkpoint runs/metr_la_dgcdiff/best.pth
```

Evaluation draws stochastic forecasts without exposing future target values, and writes aggregate and per-window  
metrics under `runs/metr_la_dgcdiff/evaluation/`.

## Source layout

- `train.py` and `src/training/`: training, checkpoints, and configuration
- `evaluate.py` and `src/evaluation/`: probabilistic evaluation and metrics
- `src/data_processing/metr_la.py`: METR-LA dataset and chronological windows
- `src/models/`: DGCDiff and its required model components
- `configs/`: training and evaluation configuration
