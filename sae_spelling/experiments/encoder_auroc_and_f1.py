from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sklearn import metrics
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from sae_spelling.experiments.common import (
    EXPERIMENTS_DIR,
    SaeInfo,
    get_gemmascope_saes_info,
    load_df_or_run,
    load_gemmascope_sae,
    load_probe,
    load_probe_data_split,
)
from sae_spelling.probing import LinearProbe
from sae_spelling.vocab import LETTERS

EXPERIMENT_NAME = "encoder_auroc_and_f1"
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
    norm_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0, keepdim=True)
    # Take the topk features by cos sim between the encoder and the probe
    top_sae_feats = (
        (probe.weights.to(norm_W_enc.device) @ norm_W_enc)
        .topk(topk, dim=-1)
        .indices.cpu()
    )
    probe = probe.cpu()
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
                sae_score = sae_scores[letter_i]
                token_scores[f"score_sae_{letter}_top_{topk_i}"] = sae_score
                token_scores[f"sae_{letter}_top_{topk_i}_feat"] = int(
                    top_sae_feats_list[letter_i][topk_i]
                )
        vocab_scores.append(token_scores)
    return pd.DataFrame(vocab_scores)


def find_optimal_f1_threshold(y_true, y_scores):
    # Calculate precision-recall curve
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_scores)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # Find the threshold that gives the best F1 score
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]

    return optimal_threshold, f1_scores[optimal_threshold_idx]


def build_f1_and_auroc_df(results_df, sae_info: SaeInfo, topk: int = 5):
    aucs = []
    for letter in LETTERS:
        y = (results_df["answer_letter"] == letter).values
        pred_probe = results_df[f"score_probe_{letter}"].values
        auc_probe = metrics.roc_auc_score(y, pred_probe)
        f1_probe = metrics.f1_score(y, pred_probe > 0)
        recall_probe = metrics.recall_score(y, pred_probe > 0)
        precision_probe = metrics.precision_score(y, pred_probe > 0)
        best_f1_bias_probe, f1_probe_best = find_optimal_f1_threshold(y, pred_probe)
        recall_probe_best = metrics.recall_score(y, pred_probe > best_f1_bias_probe)
        precision_probe_best = metrics.precision_score(
            y, pred_probe > best_f1_bias_probe
        )

        auc_info = {
            "auc_probe": auc_probe,
            "f1_probe": f1_probe,
            "f1_probe_best": f1_probe_best,
            "recall_probe": recall_probe,
            "precision_probe": precision_probe,
            "recall_probe_best": recall_probe_best,
            "precision_probe_best": precision_probe_best,
            "bias_f1_probe_best": best_f1_bias_probe,
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
            best_f1_bias_sae, f1_best = find_optimal_f1_threshold(y, pred_sae)
            recall_best = metrics.recall_score(y, pred_sae > best_f1_bias_sae)
            precision_best = metrics.precision_score(y, pred_sae > best_f1_bias_sae)
            auc_info[f"f1_sae_top_{topk_i}"] = f1
            auc_info[f"recall_sae_top_{topk_i}"] = recall
            auc_info[f"precision_sae_top_{topk_i}"] = precision
            auc_info[f"f1_sae_top_{topk_i}_best"] = f1_best
            auc_info[f"recall_sae_top_{topk_i}_best"] = recall_best
            auc_info[f"precision_sae_top_{topk_i}_best"] = precision_best
            auc_info[f"bias_f1_sae_top_{topk_i}_best"] = best_f1_bias_sae
            auc_info[f"sae_top_{topk_i}_feat"] = results_df[
                f"sae_{letter}_top_{topk_i}_feat"
            ].values[0]
        aucs.append(auc_info)
    return pd.DataFrame(aucs)


@torch.inference_mode()
def load_and_run_eval_probe_and_top_sae_raw_scores(
    sae_info: SaeInfo,
    tokenizer: PreTrainedTokenizerFast,
) -> pd.DataFrame:
    sae = load_gemmascope_sae(
        layer=sae_info.layer,
        l0=sae_info.l0,
        width=sae_info.width,
    )
    probe = load_probe(task="first_letter", layer=sae_info.layer)
    eval_activations, eval_data = load_probe_data_split(
        tokenizer, task="first_letter", layer=sae_info.layer, device="cpu"
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


def run_encoder_auroc_and_f1_experiments(
    layers: list[int],
    output_dir: Path | str = EXPERIMENTS_DIR / EXPERIMENT_NAME,
    task: str = "first_letter",
    force: bool = False,
    skip_1m_saes: bool = False,
) -> dict[int, list[tuple[pd.DataFrame, SaeInfo]]]:
    output_dir = Path(output_dir)

    # TODO: handle more tasks for this evaluation
    if task != "first_letter":
        raise ValueError(f"Unsupported task: {task}")

    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

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
                            sae_info, tokenizer
                        ),
                        raw_results_path,
                        force=force,
                    )

                auroc_results_df = load_df_or_run(
                    lambda: build_f1_and_auroc_df(get_raw_results_df(), sae_info),
                    auroc_results_path,
                    force=force,
                )
                results_by_layer[layer].append((auroc_results_df, sae_info))
            pbar.update(1)
    return results_by_layer
