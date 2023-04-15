import torch

__all__ = ["get_auto_device"]


def get_auto_device():
    if torch.has_cuda:
        device = f"cuda:{torch.cuda.current_device()}"
    elif torch.has_mps:
        device = "mps"
    else:
        device = "cpu"
    return device
