from collections.abc import Generator, Sequence
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


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
