import torch
from sae_lens import SAE

from sae_spelling.util import batchify, dict_zip, flip_dict


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


def test_dict_zip_zips_dictionaries_together_based_on_common_keys():
    assert list(dict_zip({1: "a", 2: "b"}, {1: "c", 2: "d"})) == [
        (1, ("a", "c")),
        (2, ("b", "d")),
    ]


def test_dict_zip_ignores_keys_that_are_not_shared():
    assert list(dict_zip({1: "a", 2: "b"}, {2: "c", 3: "d"})) == [(2, ("b", "c"))]


def test_dict_zip_works_with_more_than_two_dicts():
    assert list(dict_zip({1: "a", 2: "b"}, {1: "c", 2: "d"}, {1: "e", 2: "f"})) == [
        (1, ("a", "c", "e")),
        (2, ("b", "d", "f")),
    ]
