# Multi-Resolution CNN

## Model

A residual CNN with parallel resolution branches and cross-resolution feature exchange. The branch outputs are fused and pooled with GAP and GeM before classification. Training consists of ImageNet pretraining followed by fine-tuning on the target task.

## Files

```text
main.py                 Training entry point
src/                    Network, data loading, and training
configs/                Pretraining and fine-tuning settings
results.csv             Experiment results
data/
  imagenet100/
    train/<class>/
    val/<class>/
  6d958be074577803/cifar-10-batches-py/
  fbc41b31d544714d/hymenoptera_data/
    train/ants/  train/bees/
    val/ants/    val/bees/
  splits/
```

Place the datasets under `data/` using this layout. Data paths are set in `configs/`.

## Run

```bash
pip install -r requirements.txt
python main.py pretrain
python main.py finetune --checkpoint outputs/pretrain/best.pt
```

## Results

| Experiment | Accuracy | Macro recall | Insect accuracy |
|---|---:|---:|---:|
| ImageNet-100 + balanced sampling | 93.61% | 92.64% | 86.93% |
| ImageNet-100 + natural sampling | 93.42% | 91.02% | 77.12% |
| ImageNet-100 + balanced sampling, no exchange | 93.06% | 92.10% | 86.93% |
| ImageNet-100 + balanced sampling, no GeM | 93.15% | 91.92% | 84.97% |
| ImageNet-1K + balanced sampling | 96.44% | 95.57% | 90.85% |
| Random initialization + balanced sampling | 91.28% | 88.15% | 71.24% |

[CSV](results.csv)
