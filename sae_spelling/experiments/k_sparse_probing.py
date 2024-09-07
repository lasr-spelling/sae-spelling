from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sae_lens import SAE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tueplots import axes, bundles

from sae_spelling.experiments.common import (
    EXPERIMENTS_DIR,
    SaeInfo,
    get_gemmascope_saes_info,
    get_task_dir,
    humanify_sae_width,
    load_df_or_run,
    load_gemmascope_sae,
    load_probe,
    load_probe_data_split,
)
from sae_spelling.experiments.encoder_auroc_and_f1 import find_optimal_f1_threshold
from sae_spelling.probing import LinearProbe, train_multi_probe
from sae_spelling.util import DEFAULT_DEVICE, batchify
from sae_spelling.vocab import LETTERS

EPS = 1e-6
KS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50)
SPARSE_PROBING_EXPERIMENT_NAME = "k_sparse_probing"


class KSparseProbe(nn.Module):
    weight: torch.Tensor  # shape (k)
    bias: torch.Tensor  # scalar
    feature_ids: torch.Tensor  # shape (k)

    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor, feature_ids: torch.Tensor
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.feature_ids = feature_ids

    @property
    def k(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered_acts = (
            x[:, self.feature_ids] if len(x.shape) == 2 else x[self.feature_ids]
        )
        return filtered_acts @ self.weight + self.bias


def train_sparse_multi_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples, num_probes), with values in [0, 1]
    l1_decay: float = 0.01,  # l1 regularization strength
    num_probes: int | None = None,  # inferred from y_train if None
    batch_size: int = 4096,
    num_epochs: int = 50,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    l2_decay: float = 1e-6,
    show_progress: bool = True,
    verbose: bool = False,
    device: torch.device = DEFAULT_DEVICE,
) -> LinearProbe:
    """
    Train a multi-probe with L1 regularization on the weights.
    """
    return train_multi_probe(
        x_train,
        y_train,
        num_probes=num_probes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        end_lr=end_lr,
        weight_decay=l2_decay,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
        extra_loss_fn=lambda probe, _x, _y: l1_decay
        * probe.weights.abs().sum(dim=-1).mean(),
    )


def _get_sae_acts(
    sae: SAE,
    input_activations: torch.Tensor,
    sae_post_act: bool,  # whether to train the probe before or after the SAE Relu activation
    batch_size: int = 4096,
) -> torch.Tensor:
    hook_point = "hook_sae_acts_post" if sae_post_act else "hook_sae_acts_pre"
    batch_acts = []
    for batch in batchify(input_activations, batch_size):
        acts = sae.run_with_cache(batch.to(sae.device))[1][hook_point].cpu()
        batch_acts.append(acts)
    return torch.cat(batch_acts)


def train_k_sparse_probes(
    sae: SAE,
    train_labels: list[tuple[str, int]],  # list of (token, letter number) pairs
    train_activations: torch.Tensor,  # n_vocab X d_model
    ks: Iterable[int] = KS,
    sae_post_act: bool = True,  # whether to train the probe before or after the SAE Relu activation
) -> dict[int, dict[int, KSparseProbe]]:  # dict[k, dict[letter_id, probe]]
    """
    Train k-sparse probes for each k in ks.
    Returns a dict of dicts, where the outer dict is indexed by k and the inner dict is the label.
    """
    results: dict[int, dict[int, KSparseProbe]] = defaultdict(dict)
    with torch.no_grad():
        labels = {label for _, label in train_labels}
        sparse_train_y = torch.nn.functional.one_hot(
            torch.tensor([idx for _, idx in train_labels])
        )
        sae_feat_acts = _get_sae_acts(sae, train_activations, sae_post_act)
    l1_probe = (
        train_sparse_multi_probe(
            sae_feat_acts.to(sae.device),
            sparse_train_y.to(sae.device),
            l1_decay=0.01,
            num_epochs=50,
            device=sae.device,
        )
        .float()
        .cpu()
    )
    with torch.no_grad():
        sae_feat_acts_np = sae_feat_acts.cpu().float().numpy()
        train_k_y = np.array([idx for _, idx in train_labels])
        for k in ks:
            for label in labels:
                # using topk and not abs() because we only want features that directly predict the label
                sparse_feat_ids = l1_probe.weights[label].topk(k).indices.numpy()
                train_k_x = sae_feat_acts_np[:, sparse_feat_ids]
                # Use SKLearn here because it's much faster than torch if the data is small
                sk_probe = LogisticRegression(
                    max_iter=500, class_weight="balanced"
                ).fit(train_k_x, (train_k_y == label).astype(np.int64))
                probe = KSparseProbe(
                    weight=torch.tensor(sk_probe.coef_[0]).float(),
                    bias=torch.tensor(sk_probe.intercept_[0]).float(),
                    feature_ids=torch.tensor(sparse_feat_ids),
                )
                results[k][label] = probe
    return results


@torch.inference_mode()
def eval_probe_and_sae_k_sparse_raw_scores(
    sae: SAE,
    probe: LinearProbe,
    k_sparse_probes: dict[int, dict[int, KSparseProbe]],
    eval_labels: list[tuple[str, int]],  # list of (token, letter number) pairs
    eval_activations: torch.Tensor,  # n_vocab X d_model
    metadata: dict[str, str | int | float] = {},
    sae_post_act: bool = True,  # whether to train the probe before or after the SAE Relu activation
) -> pd.DataFrame:
    norm_probe_weights = probe.weights / torch.norm(probe.weights, dim=-1, keepdim=True)
    norm_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0, keepdim=True)
    norm_W_dec = sae.W_dec / torch.norm(sae.W_dec, dim=-1, keepdim=True)
    probe_dec_cos = (
        (norm_probe_weights.to(norm_W_dec.device) @ norm_W_dec.T).cpu().float()
    )
    probe_enc_cos = (
        (norm_probe_weights.to(norm_W_enc.device) @ norm_W_enc).cpu().float()
    )
    probe = probe.to("cpu")

    # using a generator to avoid storing all the rows in memory
    def row_generator():
        for token_act, (token, answer_idx) in tqdm(
            zip(eval_activations, eval_labels), total=len(eval_labels)
        ):
            probe_scores = probe(token_act).tolist()
            row: dict[str, float | str | int | list[int]] = {
                "token": token,
                "answer_letter": LETTERS[answer_idx],
                **metadata,
            }
            sae_acts = (
                _get_sae_acts(sae, token_act.unsqueeze(0).to(sae.device), sae_post_act)
                .float()
                .cpu()
            ).squeeze()
            for letter_i, (letter, probe_score) in enumerate(
                zip(LETTERS, probe_scores)
            ):
                row[f"score_probe_{letter}"] = probe_score
                for k, k_probes in k_sparse_probes.items():
                    k_probe = k_probes[letter_i]
                    k_probe_score = k_probe(sae_acts)
                    sparse_acts = sae_acts[k_probe.feature_ids]
                    row[f"score_sparse_sae_{letter}_k_{k}"] = k_probe_score.item()
                    row[f"sum_sparse_sae_{letter}_k_{k}"] = sparse_acts.sum().item()
                    row[f"sparse_sae_{letter}_k_{k}_feats"] = (
                        k_probe.feature_ids.numpy()
                    )
                    row[f"sparse_sae_{letter}_k_{k}_acts"] = sparse_acts.numpy()
                    row[f"cos_probe_sae_enc_{letter}_k_{k}"] = probe_enc_cos[
                        letter_i, k_probe.feature_ids
                    ].numpy()
                    row[f"cos_probe_sae_dec_{letter}_k_{k}"] = probe_dec_cos[
                        letter_i, k_probe.feature_ids
                    ].numpy()
                    row[f"sparse_sae_{letter}_k_{k}_weights"] = (
                        k_probe.weight.float().numpy()
                    )
                    row[f"sparse_sae_{letter}_k_{k}_bias"] = k_probe.bias.item()
            yield row

    return pd.DataFrame(row_generator())


def load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
    sae_info: SaeInfo,
    tokenizer: PreTrainedTokenizerFast,
    sae_post_act: bool = True,
) -> pd.DataFrame:
    with torch.no_grad():
        sae = load_gemmascope_sae(
            layer=sae_info.layer,
            l0=sae_info.l0,
            width=sae_info.width,
        )
        probe = load_probe(task="first_letter", layer=sae_info.layer)
        train_activations, train_data = load_probe_data_split(
            tokenizer,
            task="first_letter",
            layer=sae_info.layer,
            split="train",
            device="cpu",
        )
    k_sparse_probes = train_k_sparse_probes(
        sae,
        train_data,
        train_activations,
        sae_post_act=True,
    )
    with torch.no_grad():
        eval_activations, eval_data = load_probe_data_split(
            tokenizer,
            task="first_letter",
            layer=sae_info.layer,
            split="test",
            device="cpu",
        )
        df = eval_probe_and_sae_k_sparse_raw_scores(
            sae,
            probe,
            k_sparse_probes=k_sparse_probes,
            eval_labels=eval_data,
            eval_activations=eval_activations,
            sae_post_act=sae_post_act,
            metadata={
                "layer": sae_info.layer,
                "sae_l0": sae_info.l0,
                "sae_width": sae_info.width,
                "sae_post_act": sae_post_act,
            },
        )
    return df


def build_f1_and_auroc_df(results_df, sae_info: SaeInfo):
    aucs = []
    for letter in LETTERS:
        y = (results_df["answer_letter"] == letter).values
        pred_probe = results_df[f"score_probe_{letter}"].values
        auc_probe = metrics.roc_auc_score(y, pred_probe)
        f1_probe = metrics.f1_score(y, pred_probe > 0.0)
        recall_probe = metrics.recall_score(y, pred_probe > 0.0)
        precision_probe = metrics.precision_score(y, pred_probe > 0.0)
        best_f1_bias_probe, f1_probe = find_optimal_f1_threshold(y, pred_probe)
        recall_probe_best = metrics.recall_score(y, pred_probe > best_f1_bias_probe)
        precision_probe_best = metrics.precision_score(
            y, pred_probe > best_f1_bias_probe
        )

        auc_info = {
            "auc_probe": auc_probe,
            "f1_probe": f1_probe,
            "f1_probe_best": f1_probe,
            "recall_probe": recall_probe,
            "precision_probe": precision_probe,
            "recall_probe_best": recall_probe_best,
            "precision_probe_best": precision_probe_best,
            "bias_f1_probe_best": best_f1_bias_probe,
            "letter": letter,
            "sae_l0": sae_info.l0,
            "sae_width": sae_info.width,
        }

        for k in KS:
            pred_sae = results_df[f"score_sparse_sae_{letter}_k_{k}"].values
            auc_sae = metrics.roc_auc_score(y, pred_sae)
            f1 = metrics.f1_score(y, pred_sae > 0.0)
            recall = metrics.recall_score(y, pred_sae > 0.0)
            precision = metrics.precision_score(y, pred_sae > 0.0)
            auc_info[f"auc_sparse_sae_{k}"] = auc_sae
            best_f1_bias_sae, f1_sae_best = find_optimal_f1_threshold(y, pred_sae)
            recall_sae_best = metrics.recall_score(y, pred_sae > best_f1_bias_sae)
            precision_sae_best = metrics.precision_score(y, pred_sae > best_f1_bias_sae)
            sum_sae_pred = results_df[f"sum_sparse_sae_{letter}_k_{k}"].values
            auc_sum_sae = metrics.roc_auc_score(y, sum_sae_pred)
            f1_sum_sae = metrics.f1_score(y, sum_sae_pred > EPS)
            recall_sum_sae = metrics.recall_score(y, sum_sae_pred > EPS)
            precision_sum_sae = metrics.precision_score(y, sum_sae_pred > EPS)

            auc_info[f"f1_sparse_sae_{k}"] = f1
            auc_info[f"recall_sparse_sae_{k}"] = recall
            auc_info[f"recall_sparse_sae_{k}_best"] = recall_sae_best
            auc_info[f"precision_sparse_sae_{k}"] = precision
            auc_info[f"precision_sparse_sae_{k}_best"] = precision_sae_best
            auc_info[f"f1_sparse_sae_{k}_best"] = f1_sae_best
            auc_info[f"bias_f1_sparse_sae_{k}_best"] = best_f1_bias_sae
            auc_info[f"auc_sum_sparse_sae_{k}"] = auc_sum_sae
            auc_info[f"f1_sum_sparse_sae_{k}"] = f1_sum_sae
            auc_info[f"recall_sum_sparse_sae_{k}"] = recall_sum_sae
            auc_info[f"precision_sum_sparse_sae_{k}"] = precision_sum_sae
            auc_info[f"sparse_sae_k_{k}_feats"] = results_df[
                f"sparse_sae_{letter}_k_{k}_feats"
            ].values[0]
            auc_info[f"cos_probe_sae_enc_{letter}_k_{k}"] = results_df[
                f"cos_probe_sae_enc_{letter}_k_{k}"
            ].values[0]
            auc_info[f"cos_probe_sae_dec_{letter}_k_{k}"] = results_df[
                f"cos_probe_sae_dec_{letter}_k_{k}"
            ].values[0]
            auc_info[f"sparse_sae_k_{k}_weights"] = results_df[
                f"sparse_sae_{letter}_k_{k}_weights"
            ].values[0]
            auc_info[f"sparse_sae_k_{k}_bias"] = results_df[
                f"sparse_sae_{letter}_k_{k}_bias"
            ].values[0]
        aucs.append(auc_info)
    return pd.DataFrame(aucs)


def add_feature_splits_to_auroc_f1_df(
    df: pd.DataFrame, f1_jump_threshold: float = 0.03, ks=(1, 2, 3, 4, 5)
) -> None:
    """
    If a k-sparse probe has a F1 score that increases by `f1_jump_threshold` or more from the previous k-1, consider this to be feature splitting.
    """
    split_feats_by_letter = {}
    for letter in LETTERS:
        prev_best = -100
        df_letter = df[df["letter"] == letter]
        for k in ks:
            k_score = df_letter[f"f1_sparse_sae_{k}"].iloc[0]  # type: ignore
            k_feats = df_letter[f"sparse_sae_k_{k}_feats"].iloc[0].tolist()  # type: ignore
            if k_score > prev_best + f1_jump_threshold:
                prev_best = k_score
                split_feats_by_letter[letter] = k_feats
            else:
                break
    df["split_feats"] = df["letter"].apply(
        lambda letter: split_feats_by_letter.get(letter, [])
    )
    df["num_split_features"] = df["split_feats"].apply(len) - 1


def plot_feature_splits_vs_l0(
    k_sparse_results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
    experiment_dir: Path | str = EXPERIMENTS_DIR / SPARSE_PROBING_EXPERIMENT_NAME,
    task: str = "first_letter",
):
    task_output_dir = get_task_dir(experiment_dir, task=task)

    auroc_f1_dfs = []
    for layer, result in k_sparse_results.items():
        for auroc_f1_df, sae_info in result:
            auroc_f1_df["layer"] = layer
            auroc_f1_dfs.append(auroc_f1_df)
    df = pd.concat(auroc_f1_dfs)

    df["sae_width_str"] = df["sae_width"].map(humanify_sae_width)

    sns.set_theme()
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context({**bundles.neurips2021(), **axes.lines()}):
        plt.figure(figsize=(3.75, 2.5))
        sns.scatterplot(
            df[["layer", "sae_l0", "sae_width_str", "num_split_features"]]
            .groupby(["layer", "sae_width_str", "sae_l0"])
            .mean()
            .reset_index(),
            x="sae_l0",
            y="num_split_features",
            hue="sae_width_str",
            s=15,
            rasterized=True,
        )
        plt.legend(title="SAE width", title_fontsize="small")
        plt.title("Mean feature splits per first-letter vs L0")
        plt.xlabel("L0")
        plt.ylabel("Num feature splits")
        plt.tight_layout()
        plt.savefig(task_output_dir / "feature_splitting_vs_l0.pdf")
        plt.show()


def run_k_sparse_probing_experiments(
    layers: list[int],
    experiment_dir: Path | str = EXPERIMENTS_DIR / SPARSE_PROBING_EXPERIMENT_NAME,
    task: str = "first_letter",
    sae_post_act: bool = True,
    force: bool = False,
    skip_1m_saes: bool = False,
    f1_jump_threshold: float = 0.03,
) -> dict[int, list[tuple[pd.DataFrame, SaeInfo]]]:
    task_output_dir = get_task_dir(experiment_dir, task=task)

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
                    / f"layer_{layer}_{sae_info.width}_{sae_info.l0}_post_act_{sae_post_act}_raw_results.parquet"
                )
                auroc_results_path = (
                    task_output_dir
                    / f"layer_{layer}_{sae_info.width}_{sae_info.l0}_post_act_{sae_post_act}_auroc_f1.parquet"
                )

                def get_raw_results_df():
                    return load_df_or_run(
                        lambda: load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
                            sae_info, tokenizer, sae_post_act
                        ),
                        raw_results_path,
                        force=force,
                    )

                auroc_results_df = load_df_or_run(
                    lambda: build_f1_and_auroc_df(get_raw_results_df(), sae_info),
                    auroc_results_path,
                    force=force,
                )
                add_feature_splits_to_auroc_f1_df(auroc_results_df, f1_jump_threshold)
                results_by_layer[layer].append((auroc_results_df, sae_info))
            pbar.update(1)
    return results_by_layer
