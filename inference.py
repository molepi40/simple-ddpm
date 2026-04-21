import argparse
from pathlib import Path

import torch
import torchvision

from models.ddpm import DDPM
from models.unet import UNet
from train import load_config, load_ddpm_model, load_eps_model


def normalize_state_dict_keys(state_dict: dict) -> dict:
	# torch.compile checkpoints may prefix all keys with "_orig_mod.".
	prefix = "_orig_mod."
	if not state_dict:
		return state_dict
	if all(k.startswith(prefix) for k in state_dict.keys()):
		return {k[len(prefix):]: v for k, v in state_dict.items()}
	return state_dict


def parse_labels(labels_arg: str, n_samples: int, n_classes: int, device: torch.device):
	if not labels_arg:
		return torch.randint(0, n_classes, (n_samples,), device=device, dtype=torch.long)

	values = [int(v.strip()) for v in labels_arg.split(",") if v.strip() != ""]
	if not values:
		raise ValueError("--labels is empty after parsing")

	for v in values:
		if v < 0 or v >= n_classes:
			raise ValueError(f"Label {v} is out of range [0, {n_classes - 1}]")

	labels = torch.tensor(values, device=device, dtype=torch.long)
	if labels.numel() == 1:
		labels = labels.repeat(n_samples)
	elif labels.numel() < n_samples:
		repeat = (n_samples + labels.numel() - 1) // labels.numel()
		labels = labels.repeat(repeat)[:n_samples]
	else:
		labels = labels[:n_samples]

	return labels


@torch.inference_mode()
def sample_images(
	ddpm: DDPM,
	eps_model: UNet,
	labels: torch.Tensor,
	guidance_scale: float,
	config: dict,
	device
):
	n_samples = labels.shape[0]
	
	height = config["unet_config"]["height"]
	width = config["unet_config"]["width"]
	channels = config["unet_config"]["in_channels"]

	x = torch.randn(n_samples, channels, height, width, device=device)

	use_amp = device.type == "cuda"
	for t in range(ddpm.n_steps - 1, -1, -1):
		t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
		with torch.amp.autocast("cuda", enabled=use_amp):
			eps_cond = eps_model(x, t_batch, labels)
			if guidance_scale != 1.0:
				eps_uncond = eps_model(x, t_batch, None)
				eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
			else:
				eps = eps_cond
		x = ddpm.sample_backward_step(x, t, eps, simple_var=True)

	return x


def postprocess_and_save(samples: torch.Tensor, output_path: Path, nrow: int):
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Map from [-1, 1] to [0, 1] for visualization.
	images = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
	grid = torchvision.utils.make_grid(images, nrow=nrow)
	torchvision.utils.save_image(grid, output_path)


def save_labels(labels: torch.Tensor, label_path: Path):
	label_path.parent.mkdir(parents=True, exist_ok=True)
	label_path.write_text(",".join(str(int(v)) for v in labels.tolist()) + "\n", encoding="utf-8")


def parse_args():
	parser = argparse.ArgumentParser(description="Conditional DDPM inference for MNIST checkpoint")
	parser.add_argument(
		"--config",
		type=str,
		default="./config.json",
		help="Config file for the model and training"
	)
	parser.add_argument(
		"--ckpt-path",
		type=str,
		default="/home/xhj/notes/ddpm/ddpm_unet.pt",
		help="Path to trained epsilon model checkpoint (.pt)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="/home/xhj/notes/ddpm/samples.png",
		help="Output image path",
	)
	parser.add_argument(
		"--n-samples",
		type=int,
		default=64,
		help="Number of images to generate",
	)
	parser.add_argument(
		"--labels",
		type=str,
		default="",
		help="Comma-separated labels (e.g. 3 or 0,1,2,3); empty means random labels",
	)
	parser.add_argument(
		"--guidance-scale",
		type=float,
		default=2.0,
		help="Classifier-free guidance scale, 1.0 disables CFG",
	)
	parser.add_argument(
		"--nrow",
		type=int,
		default=8,
		help="Number of images per row in saved grid",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device, e.g. cpu / cuda / cuda:0",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed",
	)
	return parser.parse_args()


def main():
    # parse args
	args = parse_args()
	torch.manual_seed(args.seed)

	# load models
	config = load_config(args.config)
	device = torch.device(args.device)
	ddpm_model = load_ddpm_model({**config["ddpm_config"], "device": device})
	eps_model = load_eps_model({**config["unet_config"], "device": device})

	# load model weigths
	ckpt_path = Path(args.ckpt_path)
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	state_dict = torch.load(ckpt_path, map_location=device)
	state_dict = normalize_state_dict_keys(state_dict)
	try:
		eps_model.load_state_dict(state_dict)
	except RuntimeError as ex:
		raise RuntimeError(
			"Checkpoint load failed after key normalization. This may be a real architecture mismatch, for example conditional vs unconditional UNet or different channel/stage settings."
		) from ex
	eps_model.eval()

	labels = parse_labels(
		labels_arg=args.labels,
		n_samples=args.n_samples,
		n_classes=config["unet_config"]["n_classes"],
		device=device,
	)
	samples = sample_images(
		ddpm=ddpm_model,
		eps_model=eps_model,
		labels=labels,
		guidance_scale=args.guidance_scale,
		config=config,
		device=device,
	)
	output_path = Path(args.output)
	postprocess_and_save(samples=samples, output_path=output_path, nrow=args.nrow)
	save_labels(labels, output_path.with_suffix(".labels.txt"))

	print(f"Saved samples to: {output_path}")
	print(f"Saved labels to: {output_path.with_suffix('.labels.txt')}")


if __name__ == "__main__":
    main()
