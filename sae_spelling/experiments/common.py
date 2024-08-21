"""
Shared helpers for experiments
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerFast

from sae_spelling.probing import LinearProbe

DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEAM_DIR = Path("/content/drive/MyDrive/Team_Joseph")
EXPERIMENTS_DIR = TEAM_DIR / "experiments"
PROBES_DIR = TEAM_DIR / "data" / "probing_data" / "gemma-2"


def dtype_to_str(dtype: torch.dtype | str) -> str:
    return str(dtype).replace("torch.", "")


def load_gemma2_model(
    dtype: torch.dtype | str = DEFAULT_DTYPE, device: str = DEFAULT_DEVICE
) -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        dtype=dtype_to_str(dtype),
        device=device,
    )


def load_gemmascope_sae(
    layer: int,
    width: str | int = "16k",
    l0: str | int = "canonical",
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> SAE:
    if isinstance(width, int):
        if width > 999_000:
            width = f"{width // 1_000_000}m"
        else:
            width = f"{width // 1000}k"

    source = (
        "gemma-scope-2b-pt-res-canonical"
        if l0 == "canonical"
        else "gemma-scope-2b-pt-res"
    )
    l0_identifier = "canonical" if l0 == "canonical" else f"average_l0_{l0}"
    sae = SAE.from_pretrained(
        source,
        f"layer_{layer}/width_{width}/{l0_identifier}",
        device=device,
    )[0].to(dtype=dtype)
    sae.fold_W_dec_norm()
    return sae


def load_probe(
    task: str = "first_letter",
    layer: int = 0,
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> LinearProbe:
    probe = torch.load(
        Path(probes_dir) / task / f"layer_{layer}" / f"{task}_probe.pth",
        map_location=device,
    ).to(dtype=dtype)
    return probe


@torch.inference_mode()
def load_probe_data_split(
    tokenizer: PreTrainedTokenizerFast,
    task: str = "first_letter",
    layer: int = 0,
    split: Literal["train", "val"] = "val",
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    np_data = np.load(
        Path(probes_dir) / task / f"layer_{layer}" / f"{task}_data.npz",
    )
    df = pd.read_csv(
        Path(probes_dir) / task / f"layer_{layer}" / f"{task}_df.csv",
        keep_default_na=False,
        na_values=[""],
    )
    activations = torch.from_numpy(np_data[f"X_{split}"]).to(device, dtype=dtype)
    labels = np_data[f"y_{split}"].tolist()
    indices: list[int] = np_data[f"{split}_idx"].tolist()
    return _parse_probe_data_split(
        tokenizer, activations, split_labels=labels, split_indices=indices, df=df
    )


def _parse_probe_data_split(
    tokenizer: PreTrainedTokenizerFast,
    split_activations: torch.Tensor,
    split_labels: list[int],
    split_indices: list[int],
    df: pd.DataFrame,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    valid_act_indices = []
    vocab_with_labels = []
    raw_tokens_with_labels = [
        (df.iloc[idx]["token"], label)
        for idx, label in zip(split_indices, split_labels)
    ]
    for idx, (token, label) in enumerate(raw_tokens_with_labels):
        # sometimes we have tokens that look like <0x6A>
        if not isinstance(token, str) or re.match(r"[\d<>]", token):
            continue
        vocab_with_labels.append((tokenizer.convert_tokens_to_string([token]), label))
        valid_act_indices.append(idx)
    activations = split_activations[valid_act_indices]
    if activations.shape[0] != len(vocab_with_labels):
        raise ValueError(
            f"Activations and vocab with labels have different lengths: "
            f"{activations.shape[0]} != {len(vocab_with_labels)}"
        )
    return activations.clone(), vocab_with_labels


@dataclass
class SaeInfo:
    l0: int
    layer: int
    width: int
    path: str


@cache
def get_gemmascope_saes_info(layer: int | None = None) -> list[SaeInfo]:
    """
    Get a list of all available Gemmascope SAEs, optionally filtering by a specific layer.
    """
    gemma_2_saes = get_pretrained_saes_directory()["gemma-scope-2b-pt-res"]
    saes = []
    for sae_name, sae_path in gemma_2_saes.saes_map.items():
        l0 = int(gemma_2_saes.expected_l0[sae_name])
        width_match = re.search(r"width_(\d+)(k|m)", sae_name)
        assert width_match is not None
        assert width_match.group(2) in ["k", "m"]
        width = int(width_match.group(1)) * 1000
        if width_match.group(2) == "m":
            width *= 1000
        layer_match = re.search(r"layer_(\d+)", sae_name)
        assert layer_match is not None
        sae_layer = int(layer_match.group(1))
        if layer is None or sae_layer == layer:
            saes.append(SaeInfo(l0, sae_layer, width, sae_path))
    return saes
