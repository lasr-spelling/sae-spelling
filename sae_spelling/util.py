from collections.abc import Generator, Sequence
from typing import TypeVar

import torch
from tqdm.autonotebook import tqdm

T = TypeVar("T")


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]


def flip_dict(d: dict[T, T]) -> dict[T, T]:
    """Flip a dictionary, i.e. {a: b} -> {b: a}"""
    return {v: k for k, v in d.items()}


def listify(item: T | list[T]) -> list[T]:
    """Convert an item or list of items to a list."""
    if isinstance(item, list):
        return item
    return [item]
