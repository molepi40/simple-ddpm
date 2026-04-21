import torch
import torch.nn as nn
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import json

from data import get_dataloader
from models.ddpm import DDPM
from models.unet import UNet


def normalize_state_dict_keys(state_dict: dict) -> dict:
    prefix = "_orig_mod."
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    assert "ddpm_config" in config
    assert "unet_config" in config
    assert "training_config" in config
    
    return config

def load_eps_model(config: dict):
    # read config
    n_steps = config["n_steps"]
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    intermediate_channels = config["intermediate_channels"]
    height = config["height"]
    width = config["width"]
    n_classes = config["n_classes"]
    pe_dim = config["pe_dim"]
    residual = config["residual"]
    device = config["device"]
    
    unet_model = UNet(
        n_steps=n_steps,
        in_channels=in_channels,
        out_channels=out_channels,
        intermediate_channels=intermediate_channels,
        height=height,
        width=width,
        n_classes=n_classes,
        pe_dim=pe_dim,
        residual=residual
    )
    unet_model.to(device=device, non_blocking=True)
    
    return unet_model

def load_ddpm_model(config: dict):
    # read config
    n_steps = config['n_steps']
    min_beta = config['min_beta']
    max_beta = config['max_beta']
    device = torch.device(config.get("device", "cpu"))

    ddpm_model = DDPM(
        n_steps=n_steps,
        min_beta=min_beta,
        max_beta=max_beta,
        device=device,
    )

    return ddpm_model


def train(ddpm: DDPM, eps_model: nn.Module, config, ckpt_path, data_root):
    # read config
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    device = config["device"]
    num_workers = config.get("num_workers", 4)
    pin_memory = config.get("pin_memory", True)
    persistent_workers = config.get("persistent_workers", True)
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    compile_model = config.get("compile_model", False)
    label_drop_prob = config.get("label_drop_prob", 0.0)
    null_label = config.get("null_label", 10)
    
    # training setting
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    if compile_model:
        eps_model = torch.compile(eps_model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in tqdm(range(n_epochs), desc="Training epochs"):
        eps_model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=True)
        
        epoch_loss = 0.0
        num_samples = 0
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # load batch
            current_batch_size = images.shape[0]
            images = images.to(device, non_blocking=pin_memory and device.type == "cuda")
            labels = labels.to(device, non_blocking=pin_memory and device.type == "cuda")

            if label_drop_prob > 0:
                drop_mask = torch.rand(current_batch_size, device=device) < label_drop_prob
                cond_labels = labels.clone()
                cond_labels[drop_mask] = null_label
            else:
                cond_labels = labels
            
            # add noise to the images
            t = torch.randint(0, n_steps, (current_batch_size,), device=device)
            eps = torch.randn_like(images, device=device)
            images_t = ddpm.sample_forward(images, t, eps)
            
            # predict the noise
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                eps_pred = eps_model(images_t, t, cond_labels)

            # Keep loss in fp32 to avoid amp dtype mismatch (pred may be fp16, target fp32).
            loss = loss_fn(eps_pred.float(), eps.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # record loss
            epoch_loss += loss.item() * current_batch_size
            num_samples += current_batch_size
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / max(1, num_samples):.4f}"
            })
        
        print(f"Finished Epoch {epoch + 1}, Average Loss: {epoch_loss / max(1, num_samples):.4f}")
        
    # If model is compiled, save original module weights for broad compatibility.
    model_to_save = eps_model._orig_mod if hasattr(eps_model, "_orig_mod") else eps_model
    torch.save(model_to_save.state_dict(), ckpt_path)


def main():
    # parse args
    parser = argparse.ArgumentParser(description="Train a DDPM noise predictor on MNIST")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.json",
        help="Config file for the model and training"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="data root for MNIST dataset"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./pt/ddpm_unet_cond_100_epochs.pt",
        help="Where to save the trained epsilon model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume epsilon model weights from --ckpt-path if it exists",
    )
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    training_config = config["training_config"]
    unet_config = config["unet_config"]
    ddpm_config = config["ddpm_config"]
    training_config["device"] = torch.device(training_config["device"])

    # load model
    ddpm = load_ddpm_model({**ddpm_config, "device": training_config["device"]})
    eps_model = load_eps_model({**unet_config, "device": training_config["device"]})

    # load weights if check point exist and resume set
    ckpt_path = Path(args.ckpt_path)
    if args.resume and ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=training_config["device"])
        state_dict = normalize_state_dict_keys(state_dict)
        eps_model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")

    # launch training
    train(ddpm=ddpm, eps_model=eps_model, config=training_config, ckpt_path=ckpt_path, data_root=args.data)


if __name__ == "__main__":
    main()