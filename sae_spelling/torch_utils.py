import torch


def track_grad(tensor: torch.Tensor) -> None:
    """wrapper around requires_grad and retain_grad"""
    tensor.requires_grad_(True)
    tensor.retain_grad()


def extract_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Extract and clone the gradient from a tensor"""
    if tensor.grad is None:
        raise ValueError("No gradient found. Did you call track_grad on this tensor?")
    return tensor.grad.detach().clone()
