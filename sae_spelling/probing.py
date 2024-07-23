from math import exp, log
from typing import Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import (
    tqdm,
)

# Autochecks if the instane is a notebook or not (fixes weird bugs in colab)
from sae_spelling.util import DEFAULT_DEVICE


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
    end_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    show_progress: bool = True,
    verbose: bool = False,
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
        end_lr=end_lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        verbose=verbose,
    )

    return probe


def train_binary_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples,), with values in [0, 1]
    batch_size: int = 256,
    num_epochs: int = 100,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    show_progress: bool = True,
    verbose: bool = False,
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
        end_lr=end_lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
    )


def _get_exponential_decay_scheduler(
    optimizer: optim.Optimizer, start_lr: float, end_lr: float, num_steps: int
) -> optim.lr_scheduler.ExponentialLR:
    gamma = exp(log(end_lr / start_lr) / num_steps)
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def _run_probe_training(
    probe: LinearProbe,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_epochs: int,
    lr: float,
    end_lr: float,
    weight_decay: float,
    show_progress: bool,
    verbose: bool,
) -> None:
    probe.train()
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _get_exponential_decay_scheduler(
        optimizer, start_lr=lr, end_lr=end_lr, num_steps=num_epochs
    )

    # history = {'epoch_loss': [], 'learning_rate': []}

    epoch_pbar = tqdm(range(num_epochs), disable=not show_progress, desc="Epochs")
    for epoch in epoch_pbar:
        epoch_sum_loss = 0
        batch_pbar = tqdm(
            loader,
            disable=not show_progress,
            leave=False,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for batch_embeddings, batch_labels in batch_pbar:
            optimizer.zero_grad()
            logits = probe(batch_embeddings)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_sum_loss += batch_loss
            batch_pbar.set_postfix({"Loss": f"{batch_loss:.8f}"})

        epoch_mean_loss = epoch_sum_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        epoch_pbar.set_postfix(
            {"Mean Loss": f"{epoch_mean_loss:.8f}", "LR": f"{current_lr:.2e}"}
        )

        if verbose:
            print(
                f"Epoch {epoch + 1}: Mean Loss: {epoch_mean_loss:.8f}, LR: {current_lr:.2e}"
            )

        scheduler.step()

    probe.eval()
