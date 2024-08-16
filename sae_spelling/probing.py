import os
from dataclasses import dataclass
from math import exp, log
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

# Autocheck if the instance is a notebook or not (fixes weird bugs in colab)
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_spelling.prompting import Formatter, SpellingPrompt, create_icl_prompt
from sae_spelling.util import DEFAULT_DEVICE, batchify
from sae_spelling.vocab import LETTERS


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
    batch_size: int = 4096,
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


def create_dataset_probe_training(
    vocab: list[str],
    formatter: Formatter,
    num_prompts_per_token: int,
    max_icl_examples: int = 10,
    answer_class_fn: Callable[[str], int] = lambda answer: LETTERS.index(
        answer.strip().lower()
    ),
) -> list[tuple[SpellingPrompt, int]]:
    """
    Create a dataset for probe training by generating prompts for each token in the given vocabulary.

    Args:
        vocab: List of tokens in the vocabulary.
        formatter: Formatter function for answers.
        num_prompts_per_token: Number of prompts to generate for each token.
        max_icl_examples: Maximum number of in-context learning examples to include.
        answer_class_fn: Function to determine the answer class from the answer string.
                         Default is to index into LETTERS for single-character answers.

    Returns:
        A list of tuples, each containing a SpellingPrompt object and the class of the answer (typically an integer 0-26).
    """
    prompts = []

    for token in tqdm(vocab, total=len(vocab), desc="Processing tokens"):
        for _ in range(num_prompts_per_token):
            prompt = create_icl_prompt(
                word=token,
                examples=vocab,
                answer_formatter=formatter,
                max_icl_examples=max_icl_examples,
            )
            answer_class = answer_class_fn(prompt.answer)

            prompts.append(
                # {"prompt": prompt_text, "answer": answer, "answer_class": answer_class}
                (prompt, answer_class)
            )

    return prompts


def gen_and_save_df_acts_probing(
    model,
    dataset,
    path: str,
    hook_point: str,
    task_name: str,
    layer: int,
    batch_size: int = 64,
    position_idx: int = -2,
):
    """
    Generate and save activations for probing tasks to the specified path

    Args:
        model: The model to use for generating activations.
        dataset: List of tuples containing SpellingPrompt objects and answer classes.
        path: Base path for saving outputs.
        hook_point: The model hook point to extract activations from.
        task_name: Name of the task for file naming.
        batch_size: Batch size for processing.
        position_idx: Index of the token position to extract activations from. Default is -2.

    Returns:
        A tuple containing the task DataFrame and memory-mapped activation tensor.

    Note:
        This function assumes all prompts in the dataset have equal token length.
        Using a fixed position_idx will only give correct results if this assumption holds.
    """
    d_model = model.cfg.d_model

    # Pre-allocate the DataFrame for speed
    task_df = pd.DataFrame(
        {
            "prompt": [prompt.base for prompt, _ in dataset],
            "HOOK_POINT": [hook_point] * len(dataset),
            "answer": [prompt.answer for prompt, _ in dataset],
            "answer_class": [answer_class for _, answer_class in dataset],
            "token": [prompt.word for prompt, _ in dataset],
        }
    )
    task_df.index.name = "index"

    # Create the directory if it doesn't exist
    layer_path = f"layer_{layer}"
    task_dir = os.path.join(path, task_name, layer_path)
    os.makedirs(task_dir, exist_ok=True)

    memmap_path = os.path.join(task_dir, f"{task_name}_act_tensor.dat")

    # Create a memmap file for act_tensor
    act_tensor_memmap = np.memmap(
        memmap_path, dtype="float32", mode="w+", shape=(len(dataset), d_model)
    )

    with torch.no_grad():
        for i, batch in enumerate(batchify(dataset, batch_size, show_progress=True)):
            batch_prompts = [prompt.base for prompt, _ in batch]
            _, cache = model.run_with_cache(batch_prompts)

            # Numpy doesn't support bfloat16, have to manually convert to fp32
            acts = cache[hook_point][:, position_idx, :].to(torch.float32).cpu().numpy()

            start_idx = i * batch_size
            end_idx = start_idx + len(batch)
            act_tensor_memmap[start_idx:end_idx] = acts

    # Flush the memmap to ensure all data is written to disk
    act_tensor_memmap.flush()

    # Save the df to disk
    df_path = os.path.join(task_dir, f"{task_name}_df.csv")
    task_df.to_csv(df_path, index=True)

    return task_df, act_tensor_memmap


def load_df_acts_probing(
    model: HookedTransformer, path: str, task: str, layer: int
) -> tuple[pd.DataFrame, np.memmap]:
    """
    Load the DataFrame and activation tensor for a specific probing task.

    Args:
        model: The model used for generating activations (needed for d_model).
        path: Base path where the data is stored.
        task: Name of the task for file naming.

    Returns:
        A tuple containing the task DataFrame and memory-mapped activation tensor.

    Raises:
        FileNotFoundError: If the CSV or memory-mapped file is not found.
        ValueError: If there's a mismatch between DataFrame and activation tensor sizes.
    """
    d_model = model.cfg.d_model
    layer_path = f"layer_{layer}"

    df_path = os.path.join(path, task, layer_path, f"{task}_df.csv")
    act_path = os.path.join(path, task, layer_path, f"{task}_act_tensor.dat")

    try:
        task_df = pd.read_csv(df_path, index_col="index")
    except FileNotFoundError:
        raise FileNotFoundError(f"DataFrame file not found at {df_path}")

    try:
        task_acts = np.memmap(
            act_path, shape=(len(task_df), d_model), mode="r", dtype="float32"
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Activation tensor file not found at {act_path}")

    # Verify that DataFrame and activation tensor sizes match
    if len(task_df) != task_acts.shape[0]:
        raise ValueError(
            f"Mismatch between DataFrame length ({len(task_df)}) and activation tensor first dimension ({task_acts.shape[0]})"
        )

    return task_df, task_acts


def train_linear_probe_for_task(
    task_df: pd.DataFrame,
    device: torch.device,
    task_activations: np.memmap,
    num_classes: int = 26,
    batch_size: int = 4096,
    num_epochs: int = 50,
    lr: float = 1e-2,
    test_size: float = 0.2,
    random_state: int = 42,
    weight_decay=1e-4,
    use_stratify=True,
):
    y = np.array(task_df["answer_class"].values)
    original_indices = task_df.index.values  # Save the original indices

    if use_stratify:
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            task_activations,
            y,
            original_indices,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    else:
        X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
            task_activations,
            y,
            original_indices,
            test_size=test_size,
            random_state=random_state,
        )

    X_train_tensor = torch.from_numpy(X_train).float().to(device).requires_grad_(True)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device)
    y_train_one_hot = one_hot(y_train_tensor, num_classes=num_classes)

    probe = train_multi_probe(
        x_train=X_train_tensor.detach().clone(),
        y_train=y_train_one_hot.detach().clone(),
        num_probes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        show_progress=True,
        verbose=False,
        device=device,
    )

    probe_data = {
        "X_train": X_train_tensor,
        "X_val": X_val_tensor,
        "y_train": y_train_tensor,
        "y_val": y_val_tensor,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }

    return probe, probe_data


@dataclass
class ProbeStats:
    letter: str
    f1: float
    accuracy: float
    precision: float
    recall: float


def gen_probe_stats(
    probe: LinearProbe,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device = DEFAULT_DEVICE,
) -> list[ProbeStats]:
    """
    Generate statistics for a trained probe on validation data,
    treating each letter independently.

    Args:
        probe: The trained LinearProbe.
        X_val: Validation input tensor.
        y_val: Validation target tensor.
        device: The device to run computations on (default: CUDA if available, else CPU).

    Returns:
        A list of ProbeStats objects containing performance metrics for each letter.
    """

    def validator_fn(x: torch.Tensor) -> torch.Tensor:
        logits = probe(x.clone().detach().to(device))
        return (logits > threshold).float().cpu()

    results: list[ProbeStats] = []

    val_preds = validator_fn(X_val)
    y_val_cpu = y_val.cpu()

    for i, letter in enumerate(LETTERS):
        letter_preds = val_preds[:, i]
        letter_val_y = y_val_cpu == i
        results.append(
            ProbeStats(
                letter=letter,
                f1=float(f1_score(letter_val_y, letter_preds, average="binary")),
                accuracy=float(accuracy_score(letter_val_y, letter_preds)),
                precision=float(
                    precision_score(letter_val_y, letter_preds, average="binary")
                ),
                recall=float(
                    recall_score(letter_val_y, letter_preds, average="binary")
                ),
            )
        )
    return results
