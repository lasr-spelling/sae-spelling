from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sae_lens import SAE
from sklearn import metrics
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tueplots import axes, bundles

from sae_spelling.experiments.common import (
    EXPERIMENTS_DIR,
    PROBES_DIR,
    SaeInfo,
    get_gemmascope_saes_info,
    get_or_make_dir,
    humanify_sae_width,
    load_df_or_run,
    load_gemmascope_sae,
    load_probe,
    load_probe_data_split,
)
from sae_spelling.probing import LinearProbe
from sae_spelling.vocab import LETTERS

LATENT_EVALUATION_EXPERIMENT_NAME = "latent_evaluation"
EPS = 1e-8


@torch.inference_mode()
def eval_probe_and_top_sae_raw_scores(
    sae: SAE,
    probe: LinearProbe,
    eval_labels: list[tuple[str, int]],  # list of (token, letter number) pairs
    eval_activations: torch.Tensor,  # n_vocab X d_model
    topk: int = 5,
    metadata: dict[str, str | int | float] = {},
) -> pd.DataFrame:
    norm_probe_weights = probe.weights / torch.norm(probe.weights, dim=-1, keepdim=True)
    norm_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0, keepdim=True)
    norm_W_dec = sae.W_dec / torch.norm(sae.W_dec, dim=-1, keepdim=True)
    probe_dec_cos = (norm_probe_weights.to(norm_W_dec.device) @ norm_W_dec.T).cpu()
    probe_enc_cos = (norm_probe_weights.to(norm_W_enc.device) @ norm_W_enc).cpu()
    # Take the topk features by cos sim between the encoder and the probe
    top_sae_feats = probe_enc_cos.topk(topk, dim=-1).indices
    probe = probe.cpu()
    effective_bias = sae.b_enc
    # jumprelu SAEs have a separate threshold which must be passed before a feature can fire
    if hasattr(sae, "threshold"):
        effective_bias = sae.b_enc - sae.threshold
    top_feat_weights = [sae.W_enc.T[top_sae_feats[:, i]].cpu() for i in range(topk)]
    top_feat_biases = [effective_bias[top_sae_feats[:, i]].cpu() for i in range(topk)]

    top_sae_feats_list = top_sae_feats.tolist()

    vocab_scores = []
    for token_act, (token, answer_idx) in tqdm(
        zip(eval_activations, eval_labels), total=len(eval_labels)
    ):
        sae_scores_topk = [
            (token_act @ top_feat_weights[i].T + top_feat_biases[i]).tolist()
            for i in range(topk)
        ]
        probe_scores = probe(token_act).tolist()
        token_scores: dict[str, float | str | int] = {
            "token": token,
            "answer_letter": LETTERS[answer_idx],
            **metadata,
        }
        for letter_i, (letter, probe_score) in enumerate(zip(LETTERS, probe_scores)):
            token_scores[f"score_probe_{letter}"] = probe_score
            for topk_i, sae_scores in enumerate(sae_scores_topk):
                feat_id = int(top_sae_feats_list[letter_i][topk_i])
                sae_score = sae_scores[letter_i]
                token_scores[f"score_sae_{letter}_top_{topk_i}"] = sae_score
                token_scores[f"sae_{letter}_top_{topk_i}_feat"] = feat_id
                token_scores[f"cos_probe_sae_enc_{letter}_top_{topk_i}"] = (
                    probe_enc_cos[letter_i, feat_id].item()
                )
                token_scores[f"cos_probe_sae_dec_{letter}_top_{topk_i}"] = (
                    probe_dec_cos[letter_i, feat_id].item()
                )
        vocab_scores.append(token_scores)
    return pd.DataFrame(vocab_scores)


def build_evaluation_df(results_df, sae_info: SaeInfo, topk: int = 5):
    aucs = []
    for letter in LETTERS:
        y = (results_df["answer_letter"] == letter).values
        pred_probe = results_df[f"score_probe_{letter}"].values
        auc_probe = metrics.roc_auc_score(y, pred_probe)
        f1_probe = metrics.f1_score(y, pred_probe > 0)
        recall_probe = metrics.recall_score(y, pred_probe > 0)
        precision_probe = metrics.precision_score(y, pred_probe > 0)

        auc_info = {
            "auc_probe": auc_probe,
            "f1_probe": f1_probe,
            "recall_probe": recall_probe,
            "precision_probe": precision_probe,
            "letter": letter,
            "sae_l0": sae_info.l0,
            "sae_width": sae_info.width,
        }

        for topk_i in range(topk):
            pred_sae = results_df[f"score_sae_{letter}_top_{topk_i}"].values
            auc_sae = metrics.roc_auc_score(y, pred_sae)
            f1 = metrics.f1_score(y, pred_sae > EPS)
            recall = metrics.recall_score(y, pred_sae > EPS)
            precision = metrics.precision_score(y, pred_sae > EPS)
            auc_info[f"auc_sae_top_{topk_i}"] = auc_sae
            auc_info[f"f1_sae_top_{topk_i}"] = f1
            auc_info[f"recall_sae_top_{topk_i}"] = recall
            auc_info[f"precision_sae_top_{topk_i}"] = precision
            auc_info[f"sae_top_{topk_i}_feat"] = results_df[
                f"sae_{letter}_top_{topk_i}_feat"
            ].values[0]
            auc_info[f"cos_probe_sae_enc_{letter}_top_{topk_i}"] = results_df[
                f"cos_probe_sae_enc_{letter}_top_{topk_i}"
            ].values[0]
            auc_info[f"cos_probe_sae_dec_{letter}_top_{topk_i}"] = results_df[
                f"cos_probe_sae_dec_{letter}_top_{topk_i}"
            ].values[0]
        aucs.append(auc_info)
    return pd.DataFrame(aucs)


@torch.inference_mode()
def load_and_run_eval_probe_and_top_sae_raw_scores(
    sae_info: SaeInfo,
    tokenizer: PreTrainedTokenizerFast,
    probes_dir: Path | str,
) -> pd.DataFrame:
    sae = load_gemmascope_sae(
        layer=sae_info.layer,
        l0=sae_info.l0,
        width=sae_info.width,
    )
    probe = load_probe(layer=sae_info.layer, probes_dir=probes_dir)
    eval_activations, eval_data = load_probe_data_split(
        tokenizer,
        layer=sae_info.layer,
        device="cpu",
        probes_dir=probes_dir,
    )
    df = eval_probe_and_top_sae_raw_scores(
        sae,
        probe,
        eval_data,
        eval_activations,
        metadata={
            "layer": sae_info.layer,
            "sae_l0": sae_info.l0,
            "sae_width": sae_info.width,
        },
    )
    return df


def _consolidate_results_df(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
) -> pd.DataFrame:
    auroc_f1_dfs = []
    for layer, result in results.items():
        for auroc_f1_df, sae_info in result:
            auroc_f1_df["layer"] = layer
            auroc_f1_dfs.append(auroc_f1_df)
    df = pd.concat(auroc_f1_dfs)
    df["sae_width_str"] = df["sae_width"].map(humanify_sae_width)
    return df


def plot_metric_vs_l0(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
    metric: Literal["f1", "precision", "recall"] = "f1",
    experiment_dir: Path | str = EXPERIMENTS_DIR / LATENT_EVALUATION_EXPERIMENT_NAME,
    layers_range: tuple[int, int] | None = None,
):
    task_output_dir = get_or_make_dir(experiment_dir)
    df = _consolidate_results_df(results)

    title = f"First-letter SAE {metric} vs L0"
    save_file_base = f"{metric}_vs_l0"
    if layers_range is not None:
        df = df[df["layer"].between(*layers_range)]
        title += f" (layers {layers_range[0]}-{layers_range[1] - 1})"
        save_file_base += f"_layers_{layers_range[0]}_{layers_range[1] - 1}"

    sns.set_theme()
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context({**bundles.neurips2021(), **axes.lines()}):
        plt.figure(figsize=(3.75, 2.5))
        sns.scatterplot(
            df[["layer", "sae_l0", "sae_width_str", f"{metric}_sae_top_0"]]
            .groupby(["layer", "sae_width_str", "sae_l0"])  # type: ignore
            .mean()
            .reset_index(),
            x="sae_l0",
            y=f"{metric}_sae_top_0",
            hue="sae_width_str",
            s=15,
            rasterized=True,
        )
        plt.legend(title="SAE width", title_fontsize="small")
        plt.title(title)
        plt.xlabel("L0")
        plt.ylabel(f"Mean {metric}")
        plt.tight_layout()
        plt.savefig(task_output_dir / f"{save_file_base}.pdf")
        plt.show()


def plot_metric_vs_layer(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
    metric: Literal["f1", "precision", "recall"] = "f1",
    experiment_dir: Path | str = EXPERIMENTS_DIR / LATENT_EVALUATION_EXPERIMENT_NAME,
):
    task_output_dir = get_or_make_dir(experiment_dir)
    df = _consolidate_results_df(results)

    grouped_df = (
        df[["layer", "sae_l0", "sae_width_str", f"{metric}_sae_top_0"]]
        .groupby(["layer", "sae_width_str", "sae_l0"])
        .mean()
        .reset_index()
    )
    probe_df = df[["layer", "f1_probe"]].groupby(["layer"]).mean().reset_index()

    sns.set_theme()
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context({**bundles.neurips2021(), **axes.lines()}):
        plt.figure(figsize=(3.75, 2.5))
        sns.swarmplot(
            data=grouped_df,
            x="layer",
            y=f"{metric}_sae_top_0",
            hue="sae_width_str",
            size=2,
        )

        # Add the line plot for probe_df
        sns.lineplot(
            data=probe_df,
            x="layer",
            y="f1_probe",
            color="gray",
            linewidth=1,
            marker="o",
            markersize=3,
            label="Probe",
        )

        current_ticks = plt.gca().get_xticks()
        if len(current_ticks) > 20:
            # Select every other tick
            current_labels = [int(tick) for tick in current_ticks]
            new_ticks = current_ticks[::2]
            new_labels = current_labels[::2]
            plt.xticks(new_ticks, new_labels)  # type: ignore

        # Customize the plot
        plt.title(f"First-letter SAE {metric} vs Layer")
        plt.xlabel("Layer")
        plt.ylabel(f"Mean {metric}")
        plt.legend(
            title="SAE width",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title_fontsize="small",
        )

        # Adjust layout to prevent clipping of the legend
        plt.tight_layout()
        plt.savefig(task_output_dir / f"{metric}_vs_layer.pdf")
        plt.show()


def run_latent_evaluation_experiments(
    layers: list[int],
    experiment_dir: Path | str = EXPERIMENTS_DIR / LATENT_EVALUATION_EXPERIMENT_NAME,
    probes_dir: Path | str = PROBES_DIR,
    force: bool = False,
    skip_1m_saes: bool = True,
    skip_32k_saes: bool = True,
    skip_262k_saes: bool = True,
    skip_524k_saes: bool = True,
) -> dict[int, list[tuple[pd.DataFrame, SaeInfo]]]:
    task_output_dir = get_or_make_dir(experiment_dir)

    results_by_layer: dict[int, list[tuple[pd.DataFrame, SaeInfo]]] = defaultdict(list)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b"
    )  # type: ignore
    with tqdm(total=len(layers)) as pbar:
        for layer in layers:
            pbar.set_description(f"Layer {layer}")
            results_by_layer[layer] = []
            sae_infos = get_gemmascope_saes_info(layer)
            for sae_info in sae_infos:
                if skip_1m_saes and sae_info.width == 1_000_000:
                    continue
                if skip_32k_saes and sae_info.width == 32_000:
                    continue
                if skip_262k_saes and sae_info.width == 262_000:
                    continue
                if skip_524k_saes and sae_info.width == 524_000:
                    continue
                raw_results_path = (
                    task_output_dir
                    / f"layer_{layer}_{sae_info.width}_{sae_info.l0}_raw_results.parquet"
                )
                auroc_results_path = (
                    task_output_dir
                    / f"layer_{layer}_{sae_info.width}_{sae_info.l0}_auroc_f1.parquet"
                )

                def get_raw_results_df():
                    return load_df_or_run(
                        lambda: load_and_run_eval_probe_and_top_sae_raw_scores(
                            sae_info, tokenizer, probes_dir
                        ),
                        raw_results_path,
                        force=force,
                    )

                auroc_results_df = load_df_or_run(
                    lambda: build_evaluation_df(get_raw_results_df(), sae_info),
                    auroc_results_path,
                    force=force,
                )
                results_by_layer[layer].append((auroc_results_df, sae_info))
            pbar.update(1)
    return results_by_layer
