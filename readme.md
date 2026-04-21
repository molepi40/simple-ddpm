# DDPM (MNIST) - Simple Training and Inference

This is a minimal DDPM project for MNIST with a conditional UNet noise predictor.

## What This Project Does

- Trains a DDPM epsilon model on MNIST (`train.py`)
- Supports conditional generation by class label
- Generates image grids with classifier-free guidance (`inference.py`)

## Project Layout

- `train.py`: training entrypoint
- `inference.py`: sampling/inference entrypoint
- `data.py`: MNIST dataloader (`[-1, 1]` normalization)
- `models/`: DDPM and UNet implementation
- `config.json`: model and training config
- `pt/`: checkpoints
- `samples/` or custom output path: generated sample images

## Environment

Recommended:

- Python 3.10+
- PyTorch with CUDA (for GPU training/inference)

Install core dependencies in your virtual environment:

```bash
pip install torch torchvision tqdm
```

## Training

Run training with the config file:

```bash
python train.py \
	--config ./config.json \
	--data ./data \
	--ckpt-path ./pt/ddpm_unet_cond_100_epochs.pt
```

Resume from an existing checkpoint:

```bash
python train.py \
	--config ./config.json \
	--data ./data \
	--ckpt-path ./pt/ddpm_unet_cond_100_epochs.pt \
	--resume
```

## Inference

Example command:

```bash
python inference.py \
	--config ./config.json \
	--ckpt-path ./pt/ddpm_unet_cond_100_epochs.pt \
	--output ./samples_cond_6.png \
	--n-samples 8 \
	--nrow 4 \
	--labels 2 \
	--guidance-scale 2.5 \
	--device cuda:0
```

Notes:

- `--labels` can be a single label (`2`) or comma-separated list (`0,1,2,3`)
- If fewer labels are provided than `--n-samples`, labels are repeated
- A label file is saved next to output image (`*.labels.txt`)

## Reference

- [Diffusion Model 详解：直观理解、数学原理、PyTorch 实现](https://zhuanlan.zhihu.com/p/638442430)

