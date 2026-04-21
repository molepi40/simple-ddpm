import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Lambda
from pathlib import Path

def get_dataloader(
    root: Path,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root=root, download=True, transform=transform)

    use_workers = max(0, int(num_workers))
    use_persistent_workers = bool(persistent_workers and use_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=use_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )