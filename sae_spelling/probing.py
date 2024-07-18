import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from sae_spelling.util import DEFAULT_DEVICE


class LinearProbe(nn.Module):
    """
    Based on by https://github.com/jbloomAus/alphabetical_probe/blob/main/src/probes.py
    """

    def __init__(self, input_dim, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)


def get_class_weights(y, num_classes):
    """
    Returns a tensor of inverse class frequencies
    """
    return (len(y) / torch.bincount(y, minlength=num_classes)).tolist()


def create_rebalanced_dataloader(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    batch_size: int = 512,
):
    """
    Create a dataloader with weighted sampling based on class frequencies
    """
    dataset = TensorDataset(x, y)
    class_weights = get_class_weights(y, num_classes)
    sampler = WeightedRandomSampler([class_weights[label] for label in y], len(y))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_multiclass_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples,), with values in [0, num_classes)
    num_classes: int | None = None,  # inferred from y_train if None
    batch_size: int = 256,
    num_epochs: int = 5,
    lr: float = 0.001,
    weight_decay: float = 1e-7,
    show_progress: bool = True,
    device: torch.device = DEFAULT_DEVICE,
) -> LinearProbe:
    """
    Train a logistic regression probe on the given data

    Args:
        x_train: tensor of shape (num_samples, input_dim)
        y_train: tensor of shape (num_samples,), with values in [0, num_classes)
        num_classes: number of classes in the dataset
        batch_size: batch size for training
        num_epochs: number of epochs to train for
        lr: learning rate
        weight_decay: weight decay
        show_progress: whether to show a progress bar
        device: device to train on
    """
    num_classes = num_classes or len(torch.unique(y_train))
    loader = create_rebalanced_dataloader(
        x_train.to(device), y_train.to(device), num_classes, batch_size=batch_size
    )
    model = LinearProbe(x_train.shape[-1], num_classes=num_classes)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        pbar = tqdm(loader, disable=not show_progress)
        epoch_sum_loss = 0
        for batch_embeddings, batch_labels in pbar:
            optimizer.zero_grad()
            logits = model(batch_embeddings)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_sum_loss += loss.item()
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}"
            )
        epoch_mean_loss = epoch_sum_loss / len(loader)
        scheduler.step(epoch_mean_loss)

    model.eval()

    return model
