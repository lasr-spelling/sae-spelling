"""
Shared helpers for experiments
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerFast

from sae_spelling.probing import (
    LinearProbe,
    create_dataset_probe_training,
    gen_and_save_df_acts_probing,
    save_probe_and_data,
    train_linear_probe_for_task,
)
from sae_spelling.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    Formatter,
    first_letter_formatter,
)
from sae_spelling.vocab import get_alpha_tokens

DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENTS_DIR = Path.cwd() / "experiments"
# TODO: add probe training code
PROBES_DIR = Path.cwd() / "probes"


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


def load_or_train_probe(
    model: HookedTransformer,
    layer: int = 0,
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> LinearProbe:
    probe_path = Path(probes_dir) / f"layer_{layer}" / "probe.pth"
    if not probe_path.exists():
        print(f"Probe for layer {layer} not found, training...")
        train_and_save_probes(
            model,
            [layer],
            probes_dir,
        )
    return load_probe(layer, probes_dir, dtype, device)


def load_probe(
    layer: int = 0,
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> LinearProbe:
    probe = torch.load(
        Path(probes_dir) / f"layer_{layer}" / "probe.pth",
        map_location=device,
    ).to(dtype=dtype)
    return probe


def load_probe_data_split_or_train(
    model: HookedTransformer,
    layer: int = 0,
    split: Literal["train", "test"] = "test",
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    probe_path = Path(probes_dir) / f"layer_{layer}" / "probe.pth"
    if not probe_path.exists():
        print(f"Probe for layer {layer} not found, training...")
        train_and_save_probes(
            model,
            [layer],
            probes_dir,
        )
    return load_probe_data_split(
        model.tokenizer,  # type: ignore
        layer,
        split,
        probes_dir,
        dtype,
        device,
    )


@torch.inference_mode()
def load_probe_data_split(
    tokenizer: PreTrainedTokenizerFast,
    layer: int = 0,
    split: Literal["train", "test"] = "test",
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    np_data = np.load(
        Path(probes_dir) / f"layer_{layer}" / "data.npz",
    )
    df = pd.read_csv(
        Path(probes_dir) / f"layer_{layer}" / f"{split}_df.csv",
        keep_default_na=False,
        na_values=[""],
    )
    activations = torch.from_numpy(np_data[f"X_{split}"]).to(device, dtype=dtype)
    labels = np_data[f"y_{split}"].tolist()
    return _parse_probe_data_split(tokenizer, activations, split_labels=labels, df=df)


def _parse_probe_data_split(
    tokenizer: PreTrainedTokenizerFast,
    split_activations: torch.Tensor,
    split_labels: list[int],
    df: pd.DataFrame,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    valid_act_indices = []
    vocab_with_labels = []
    raw_tokens_with_labels = [
        (df.iloc[idx]["token"], label) for idx, label in enumerate(split_labels)
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
        # this SAE is missing, see https://github.com/jbloomAus/SAELens/pull/293. Just skip it.
        if layer == 11 and l0 == 79:
            continue
        if layer is None or sae_layer == layer:
            saes.append(SaeInfo(l0, sae_layer, width, sae_path))
    return saes


def get_or_make_dir(
    experiment_dir: str | Path,
) -> Path:
    """
    Helper to create a directory for a specific task within an experiment directory.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def load_experiment_df(
    experiment_name: str,
    path: Path,
) -> pd.DataFrame:
    """
    Helper to load a DF or error if it doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the {experiment_name} experiment first."
        )
    return pd.read_parquet(path)


def load_df_or_run(
    fn: Callable[[], pd.DataFrame],
    path: Path,
    force: bool = False,
) -> pd.DataFrame:
    return load_dfs_or_run(lambda: [fn()], [path], force)[0]


def load_dfs_or_run(
    fn: Callable[[], Iterable[pd.DataFrame]],
    paths: Iterable[Path],
    force: bool = False,
) -> list[pd.DataFrame]:
    if force or not all(path.exists() for path in paths):
        dfs = fn()
        for df, path in zip(dfs, paths):
            df.to_parquet(path, index=False)
    else:
        print(f"{paths} exist(s), loading from disk")
        dfs = [pd.read_parquet(path) for path in paths]
    return list(dfs)


def humanify_sae_width(width: int) -> str:
    """
    A helper to convert SAE width to a nicer human-readable string.
    """
    if width == 1_000_000:
        return "1m"
    else:
        return f"{width // 1_000}k"


def create_and_train_probe(
    model: HookedTransformer,
    formatter: Formatter,
    hook_point: str,
    probes_dir: str | Path,
    vocab: list[str],
    batch_size: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    base_template: str,
    pos_idx: int,
    num_prompts_per_token: int = 1,
):
    train_dataset, test_dataset = create_dataset_probe_training(
        vocab=vocab,
        formatter=formatter,
        num_prompts_per_token=num_prompts_per_token,
        base_template=base_template,
    )

    layer = int(hook_point.split(".")[1])

    train_df, test_df, train_activations, test_activations = (
        gen_and_save_df_acts_probing(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            path=probes_dir,
            hook_point=hook_point,
            batch_size=batch_size,
            layer=layer,
            position_idx=pos_idx,
        )
    )

    num_classes = 26
    probe, probe_data = train_linear_probe_for_task(
        train_df=train_df,
        test_df=test_df,
        train_activations=train_activations,
        test_activations=test_activations,
        num_classes=num_classes,
        batch_size=32 * batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )

    save_probe_and_data(probe, probe_data, probes_dir, layer)
    print("Probe saved successfully.\n")


def train_and_save_probes(
    model: HookedTransformer,
    layers: list[int],
    probes_dir: str | Path = PROBES_DIR,
    batch_size=64,
    num_epochs=50,
    lr=1e-2,
    device=torch.device("cuda"),
):
    vocab = get_alpha_tokens(model.tokenizer)  # type: ignore
    for layer in tqdm(layers):
        hook_point = f"blocks.{layer}.hook_resid_post"
        create_and_train_probe(
            model=model,
            hook_point=hook_point,
            formatter=first_letter_formatter(),
            probes_dir=probes_dir,
            vocab=vocab,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
            pos_idx=VERBOSE_FIRST_LETTER_TOKEN_POS,
        )
