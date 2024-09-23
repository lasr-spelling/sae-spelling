import pandas as pd
import torch
from syrupy.assertion import SnapshotAssertion
from transformers import PreTrainedTokenizerFast

from sae_spelling.experiments.common import (
    SaeInfo,
    _parse_probe_data_split,
    get_gemmascope_saes_info,
)


def test_get_gemmascope_saes_info(snapshot: SnapshotAssertion):
    saes = get_gemmascope_saes_info()
    assert saes == snapshot


def test_get_gemmascope_saes_info_at_specific_layer():
    l1_saes = get_gemmascope_saes_info(layer=1)
    assert l1_saes == [
        SaeInfo(l0=10, layer=1, width=16_000, path="layer_1/width_16k/average_l0_10"),
        SaeInfo(l0=102, layer=1, width=16_000, path="layer_1/width_16k/average_l0_102"),
        SaeInfo(l0=20, layer=1, width=16_000, path="layer_1/width_16k/average_l0_20"),
        SaeInfo(l0=250, layer=1, width=16_000, path="layer_1/width_16k/average_l0_250"),
        SaeInfo(l0=40, layer=1, width=16_000, path="layer_1/width_16k/average_l0_40"),
        SaeInfo(l0=121, layer=1, width=65_000, path="layer_1/width_65k/average_l0_121"),
        SaeInfo(l0=16, layer=1, width=65_000, path="layer_1/width_65k/average_l0_16"),
        SaeInfo(l0=30, layer=1, width=65_000, path="layer_1/width_65k/average_l0_30"),
        SaeInfo(l0=54, layer=1, width=65_000, path="layer_1/width_65k/average_l0_54"),
        SaeInfo(l0=9, layer=1, width=65_000, path="layer_1/width_65k/average_l0_9"),
    ]


def test_parse_probe_data_split(gpt2_tokenizer: PreTrainedTokenizerFast):
    split_activations = torch.randn(4, 12)
    split_labels = [5, 1, 18, 22]
    df = pd.DataFrame(
        {
            "token": ["fish", "bird", "shark", "whale"],
        }
    )
    activations, vocab_with_labels = _parse_probe_data_split(
        gpt2_tokenizer, split_activations, split_labels, df
    )
    assert torch.allclose(activations, split_activations)
    assert vocab_with_labels == [
        ("fish", 5),
        ("bird", 1),
        ("shark", 18),
        ("whale", 22),
    ]


def test_parse_probe_data_split_removes_invalid_rows(
    gpt2_tokenizer: PreTrainedTokenizerFast,
):
    split_activations = torch.randn(5, 12)
    split_labels = [5, 1, 18, 22, 23]
    df = pd.DataFrame(
        {
            "token": [
                "fish",
                "bird",
                float("nan"),
                "whale",
                "<0x6A>",
            ],
        }
    )
    activations, vocab_with_labels = _parse_probe_data_split(
        gpt2_tokenizer, split_activations, split_labels, df
    )
    assert torch.allclose(activations, split_activations[[0, 1, 3]])
    assert vocab_with_labels == [
        ("fish", 5),
        ("bird", 1),
        ("whale", 22),
    ]


def test_parse_probe_data_split_replaces_special_token_chars(
    gpt2_tokenizer: PreTrainedTokenizerFast,
):
    split_activations = torch.randn(2, 12)
    split_labels = [18, 22]
    df = pd.DataFrame(
        {
            "token": [
                "Ġsculpt",
                "whale",
            ],
        }
    )
    activations, vocab_with_labels = _parse_probe_data_split(
        gpt2_tokenizer, split_activations, split_labels, df
    )
    assert torch.allclose(activations, split_activations)
    assert vocab_with_labels == [
        (" sculpt", 18),
        ("whale", 22),
    ]
