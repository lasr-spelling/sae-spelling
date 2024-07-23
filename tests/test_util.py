import torch
from sae_lens import SAE

from sae_spelling.util import batchify, flip_dict


def test_batchify_splits_sequences_into_chunks():
    batches = [batch for batch in batchify(list(range(10)), batch_size=3)]
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def set_sae_dtype(sae: SAE, dtype: torch.dtype):
    sae = sae.to(dtype)
    sae.dtype = dtype
    return sae


def test_flip_dict():
    assert flip_dict({"a": "b"}) == {"b": "a"}


def test_flip_dict_with_multiple_values():
    assert flip_dict({"a": "b", "c": "d"}) == {"b": "a", "d": "c"}
