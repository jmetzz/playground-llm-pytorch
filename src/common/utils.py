import torch


def get_default_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
