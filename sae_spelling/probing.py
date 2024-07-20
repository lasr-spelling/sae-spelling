from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sae_spelling.util import DEFAULT_DEVICE

EPS = 1e-8


class LinearProbe(nn.Module):
    """
    Based on by https://github.com/jbloomAus/alphabetical_probe/blob/main/src/probes.py
    """

    def __init__(self, input_dim, num_outputs: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_outputs)

    def forward(self, x):
        return self.fc(x)

    @property
    def weights(self):
        return self.fc.weight


def _calc_pos_weights(y: torch.Tensor) -> torch.Tensor:
    num_pos_samples = y.sum(dim=0)
    num_neg_samples = len(y) - num_pos_samples
    return num_neg_samples / num_pos_samples


def train_multi_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples, num_probes), with values in [0, 1]
    num_probes: int | None = None,  # inferred from y_train if None
    batch_size: int = 256,
    num_epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-7,
    show_progress: bool = True,
    verbose: bool = False,
    early_stopping: bool = True,
    device: torch.device = DEFAULT_DEVICE,
) -> LinearProbe:
    """
    Train a multi-class one-vs-rest logistic regression probe on the given data.
    This is equivalent to training num_probes separate binary logistic regression probes.

    Args:
        x_train: tensor of shape (num_samples, input_dim)
        y_train: one_hot (or multi-hot) tensor of shape (num_samples, num_probes), with values in [0, 1]
        num_probes: number of probes to train simultaneously
        batch_size: batch size for training
        num_epochs: number of epochs to train for
        lr: learning rate
        weight_decay: weight decay
        show_progress: whether to show a progress bar
        device: device to train on
    """
    dtype = x_train.dtype
    num_probes = num_probes or y_train.shape[-1]
    dataset = TensorDataset(x_train.to(device), y_train.to(device, dtype=dtype))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    probe = LinearProbe(x_train.shape[-1], num_outputs=num_probes).to(
        device, dtype=dtype
    )

    _run_probe_training(
        probe,
        loader,
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=_calc_pos_weights(y_train)),
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        verbose=verbose,
        early_stopping=early_stopping,
    )

    return probe


def train_binary_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples,), with values in [0, 1]
    batch_size: int = 256,
    num_epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-7,
    show_progress: bool = True,
    verbose: bool = False,
    early_stopping: bool = True,
    device: torch.device = DEFAULT_DEVICE,
) -> LinearProbe:
    """
    Train a logistic regression probe on the given data. This is a thin wrapped around train_multi_probe.

    Args:
        x_train: tensor of shape (num_samples, input_dim)
        y_train: tensor of shape (num_samples,), with values in [0, 1]
        batch_size: batch size for training
        num_epochs: number of epochs to train for
        lr: learning rate
        weight_decay: weight decay
        show_progress: whether to show a progress bar
        device: device to train on
    """
    return train_multi_probe(
        x_train,
        y_train.unsqueeze(1),
        num_probes=1,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
        early_stopping=early_stopping,
    )


def _run_probe_training(
    probe: LinearProbe,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_epochs: int,
    lr: float,
    weight_decay: float,
    show_progress: bool,
    verbose: bool,
    early_stopping: bool,
) -> None:
    probe.train()
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        eps=EPS,
    )
    for epoch in range(num_epochs):
        pbar = tqdm(loader, disable=not show_progress)
        epoch_sum_loss = 0
        for batch_embeddings, batch_labels in pbar:
            optimizer.zero_grad()
            logits = probe(batch_embeddings)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_sum_loss += loss.item()
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.8f}"
            )
        epoch_mean_loss = epoch_sum_loss / len(loader)
        last_lr = scheduler.get_last_lr()
        if last_lr[0] <= 2 * EPS and early_stopping:
            if verbose:
                print("Early stopping")
            break
        if verbose:
            print(
                f"epoch {epoch} sum loss: {epoch_sum_loss:.8f}, mean loss: {epoch_mean_loss:.8f} lr: {last_lr}"
            )
        scheduler.step(epoch_mean_loss)
    probe.eval()
