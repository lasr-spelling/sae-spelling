from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformers import PreTrainedTokenizerBase
from tueplots import axes, bundles

from sae_spelling.experiments.common import (
    EXPERIMENTS_DIR,
    PROBES_DIR,
    SaeInfo,
    get_gemmascope_saes_info,
    get_task_dir,
    humanify_sae_width,
    load_df_or_run,
    load_experiment_df,
    load_gemma2_model,
    load_gemmascope_sae,
    load_probe,
)
from sae_spelling.experiments.k_sparse_probing import (
    SPARSE_PROBING_EXPERIMENT_NAME,
    add_feature_splits_to_auroc_f1_df,
    get_sparse_probing_auroc_f1_results_filename,
    get_sparse_probing_raw_results_filename,
)
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
from sae_spelling.probing import LinearProbe
from sae_spelling.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
)
from sae_spelling.vocab import LETTERS, LETTERS_UPPER, get_alpha_tokens

FEATURE_ABSORPTION_EXPERIMENT_NAME = "feature_absorption"

# the cosine similarity between the top ablation feature and the probe must be at least this high
ABSORPTION_PROBE_COS_THRESHOLD = 0.025

# the top ablation score must be at least this much larger than the second highest score
ABSORPTION_FEATURE_DELTA_THRESHOLD = 1.0


@dataclass
class StatsAndLikelyFalseNegativeResults:
    probe_true_positives: int
    split_feats_true_positives: int
    split_feats: list[int]
    potential_false_negatives: list[str]


def letter_delta_metric(tokenizer: PreTrainedTokenizerBase, pos_letter: str):
    neg_letters = [
        f" {letter}" for letter in LETTERS_UPPER if pos_letter[-1].upper() != letter
    ]
    pos_letter_tok = tokenizer.encode(pos_letter)[-1]  # type: ignore
    neg_letter_toks = torch.tensor(
        [tokenizer.encode(neg_letter)[-1] for neg_letter in neg_letters]  # type: ignore
    )

    def metric_fn(logits):
        pos_logit = logits[:, -1, pos_letter_tok]
        neg_logits = logits[:, -1, neg_letter_toks]
        result = pos_logit - (neg_logits.sum(dim=-1) / len(neg_letters))
        return result

    return metric_fn


def calculate_ig_ablation_and_cos_sims(
    calculator: FeatureAbsorptionCalculator,
    sae: SAE,
    probe: LinearProbe,
    likely_negs: dict[str, StatsAndLikelyFalseNegativeResults],
    max_prompts_per_letter: int = 200,
) -> pd.DataFrame:
    results = []
    for letter, stats in tqdm(likely_negs.items()):
        assert calculator.model.tokenizer is not None
        absorption_results = calculator.calculate_absorption_sampled(
            sae,
            words=stats.potential_false_negatives,
            probe_dir=probe.weights[LETTERS.index(letter)],
            metric_fn=letter_delta_metric(calculator.model.tokenizer, letter),
            main_feature_ids=stats.split_feats,
            max_ablation_samples=max_prompts_per_letter,
            show_progress=False,
        )
        for sample in absorption_results.sample_results:
            top_feat_score = sample.top_ablation_feature_scores[0]
            second_feat_score = sample.top_ablation_feature_scores[1]
            result = {
                "letter": letter,
                "token": sample.word,
                "prompt": sample.prompt,
                "sample_portion": absorption_results.sample_portion,
                "num_probe_true_positives": stats.probe_true_positives,
                "split_feats": stats.split_feats,
                "split_feat_acts": [
                    score.activation for score in sample.main_feature_scores
                ],
                "split_feat_probe_cos": [
                    score.probe_cos_sim for score in sample.main_feature_scores
                ],
                "top_ablation_feat": top_feat_score.feature_id,
                "top_ablation_score": top_feat_score.ablation_score,
                "top_ablation_feat_probe_cos": top_feat_score.probe_cos_sim,
                "second_ablation_feat": second_feat_score.feature_id,
                "second_ablation_score": second_feat_score.ablation_score,
                "second_ablation_feat_probe_cos": second_feat_score.probe_cos_sim,
                "ablation_scores": [
                    score.ablation_score for score in sample.top_ablation_feature_scores
                ],
                "ablation_feats": [
                    score.feature_id for score in sample.top_ablation_feature_scores
                ],
                "ablation_feat_acts": [
                    score.activation for score in sample.top_ablation_feature_scores
                ],
                "ablation_feat_probe_cos": [
                    score.probe_cos_sim for score in sample.top_ablation_feature_scores
                ],
                "is_absorption": sample.is_absorption,
            }
            results.append(result)
    result_df = pd.DataFrame(results)
    return result_df


def get_stats_and_likely_false_negative_tokens(
    auroc_f1_df: pd.DataFrame,
    sae_info: SaeInfo,
    sparse_probing_task_output_dir: Path,
) -> dict[str, StatsAndLikelyFalseNegativeResults]:
    """
    Examine the k-sparse probing results and look for false-negative cases where the k top feats don't fire but our LR probe does
    """
    results: dict[str, StatsAndLikelyFalseNegativeResults] = {}
    raw_df = load_experiment_df(
        SPARSE_PROBING_EXPERIMENT_NAME,
        sparse_probing_task_output_dir
        / get_sparse_probing_raw_results_filename(sae_info),
    )
    for letter in LETTERS:
        split_feats = auroc_f1_df[auroc_f1_df["letter"] == letter]["split_feats"].iloc(  # type: ignore
            0
        )[0]
        k = len(split_feats)
        potential_false_negatives = raw_df[
            (raw_df["answer_letter"] == letter)
            & (raw_df[f"score_probe_{letter}"] > 0)
            & (raw_df[f"score_sparse_sae_{letter}_k_{k}"] <= 0)
        ]["token"].tolist()
        num_split_feats_true_positives = raw_df[
            (raw_df["answer_letter"] == letter)
            & (raw_df[f"score_probe_{letter}"] > 0)
            & (raw_df[f"score_sparse_sae_{letter}_k_{k}"] > 0)
        ].shape[0]
        num_probe_true_positives = raw_df[
            (raw_df["answer_letter"] == letter) & (raw_df[f"score_probe_{letter}"] > 0)
        ].shape[0]
        results[letter] = StatsAndLikelyFalseNegativeResults(
            probe_true_positives=num_probe_true_positives,
            split_feats_true_positives=num_split_feats_true_positives,
            split_feats=split_feats,
            potential_false_negatives=potential_false_negatives,
        )
    return results


def load_and_run_calculate_ig_ablation_and_cos_sims(
    calculator: FeatureAbsorptionCalculator,
    auroc_f1_df: pd.DataFrame,
    sae_info: SaeInfo,
    probes_dir: Path | str,
    sparse_probing_task_output_dir: Path,
) -> pd.DataFrame:
    layer = sae_info.layer
    probe = load_probe(layer=layer, probes_dir=probes_dir)
    sae = load_gemmascope_sae(layer, width=sae_info.width, l0=sae_info.l0)
    likely_negs = get_stats_and_likely_false_negative_tokens(
        auroc_f1_df,
        sae_info,
        sparse_probing_task_output_dir,
    )
    return calculate_ig_ablation_and_cos_sims(calculator, sae, probe, likely_negs)


def _aggregate_results_df(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
) -> pd.DataFrame:
    combined = []
    for layer_results in results.values():
        for df, sae_info in layer_results:
            if "sample_portion" not in df.columns:
                df["sample_portion"] = 1.0
            agg_df = (
                df[["letter", "is_absorption"]]
                .groupby(["letter"])
                .sum()
                .reset_index()
                .merge(
                    df[["letter", "sample_portion", "num_probe_true_positives"]]
                    .groupby(["letter"])
                    .mean()
                    .reset_index()
                )
            )
            agg_df["num_absorption"] = (
                agg_df["is_absorption"] / agg_df["sample_portion"]
            )
            agg_df["absorption_rate"] = (
                agg_df["num_absorption"] / agg_df["num_probe_true_positives"]
            )
            agg_df["layer"] = sae_info.layer
            agg_df["sae_width"] = sae_info.width
            agg_df["sae_l0"] = sae_info.l0
            combined.append(agg_df)
    df = pd.concat(combined)
    df["sae_width_str"] = df["sae_width"].map(humanify_sae_width)
    return df


def plot_absorption_rate_vs_l0(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
    experiment_dir: Path | str = EXPERIMENTS_DIR / FEATURE_ABSORPTION_EXPERIMENT_NAME,
    task: str = "first_letter",
):
    task_output_dir = get_task_dir(experiment_dir, task=task)
    df = _aggregate_results_df(results)

    sns.set_theme()
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context({**bundles.neurips2021(), **axes.lines()}):
        plt.figure(figsize=(3.75, 2.5))
        sns.scatterplot(
            df[["layer", "sae_l0", "sae_width_str", "absorption_rate"]]
            .groupby(["layer", "sae_width_str", "sae_l0"])
            .mean()
            .reset_index(),
            x="sae_l0",
            y="absorption_rate",
            hue="sae_width_str",
            s=15,
            rasterized=True,
        )
        plt.legend(title="SAE width", title_fontsize="small")
        plt.title("Mean absorption rate for first-letter task vs L0")
        plt.xlabel("L0")
        plt.ylabel("Mean absorption rate")
        plt.tight_layout()
        plt.savefig(task_output_dir / "absorption_rate_vs_l0.pdf")
        plt.show()


def plot_absorption_rate_vs_layer(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
    experiment_dir: Path | str = EXPERIMENTS_DIR / FEATURE_ABSORPTION_EXPERIMENT_NAME,
    task: str = "first_letter",
):
    task_output_dir = get_task_dir(experiment_dir, task=task)
    df = _aggregate_results_df(results)
    grouped_df = (
        df[["layer", "sae_l0", "sae_width_str", "absorption_rate"]]
        .groupby(["layer", "sae_width_str", "sae_l0"])
        .mean()
        .reset_index()
    )

    sns.set_theme()
    plt.rcParams.update({"figure.dpi": 150})
    with plt.rc_context({**bundles.neurips2021(), **axes.lines()}):
        plt.figure(figsize=(3.75, 2.5))
        sns.swarmplot(
            data=grouped_df,
            x="layer",
            y="absorption_rate",
            hue="sae_width_str",
            size=2,
        )

        # Customize the plot
        plt.title("Mean absorption rate for first-letter task vs Layer")
        plt.xlabel("Layer")
        plt.ylabel("Mean absorption rate")
        plt.legend(
            title="SAE width",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title_fontsize="small",
        )

        # Adjust layout to prevent clipping of the legend
        plt.tight_layout()
        plt.savefig(task_output_dir / "absorption_rate_vs_layer.pdf")
        plt.show()


def run_feature_absortion_experiments(
    layers: list[int],
    experiment_dir: Path | str = EXPERIMENTS_DIR / FEATURE_ABSORPTION_EXPERIMENT_NAME,
    sparse_probing_experiment_dir: Path | str = EXPERIMENTS_DIR
    / SPARSE_PROBING_EXPERIMENT_NAME,
    probes_dir: Path | str = PROBES_DIR,
    task: str = "first_letter",
    force: bool = False,
    skip_1m_saes: bool = True,
    skip_32k_saes: bool = True,
    skip_262k_saes: bool = True,
    skip_524k_saes: bool = True,
    feature_split_f1_jump_threshold: float = 0.03,
    verbose: bool = True,
) -> dict[int, list[tuple[pd.DataFrame, SaeInfo]]]:
    """
    NOTE: this experiments requires the results of the k-sparse probing experiments. Make sure to run them first.
    """
    task_output_dir = get_task_dir(experiment_dir, task=task)
    sparse_probing_task_output_dir = get_task_dir(
        sparse_probing_experiment_dir, task=task
    )

    model = load_gemma2_model()
    vocab = get_alpha_tokens(model.tokenizer)  # type: ignore
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=vocab,
        max_icl_examples=10,
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        answer_formatter=first_letter_formatter(),
        word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
        probe_cos_sim_threshold=ABSORPTION_PROBE_COS_THRESHOLD,
        ablation_delta_threshold=ABSORPTION_FEATURE_DELTA_THRESHOLD,
        ig_batch_size=6,
        ig_interpolation_steps=6,
        filter_prompts_batch_size=40,
    )
    results_by_layer: dict[int, list[tuple[pd.DataFrame, SaeInfo]]] = defaultdict(list)
    with tqdm(total=len(layers)) as pbar:
        for layer in layers:
            pbar.set_description(f"Layer {layer}")
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
                if verbose:
                    print(f"Running SAE {sae_info}", flush=True)
                auroc_f1_df = load_experiment_df(
                    SPARSE_PROBING_EXPERIMENT_NAME,
                    sparse_probing_task_output_dir
                    / get_sparse_probing_auroc_f1_results_filename(sae_info),
                )
                add_feature_splits_to_auroc_f1_df(
                    auroc_f1_df, feature_split_f1_jump_threshold
                )
                df_path = (
                    task_output_dir
                    / f"layer_{layer}_width_{sae_info.width}_l0_{sae_info.l0}.parquet"
                )
                df = load_df_or_run(
                    lambda: load_and_run_calculate_ig_ablation_and_cos_sims(
                        calculator,
                        auroc_f1_df,
                        sae_info,
                        probes_dir=probes_dir,
                        sparse_probing_task_output_dir=sparse_probing_task_output_dir,
                    ),
                    df_path,
                    force=force,
                )
                results_by_layer[layer].append((df, sae_info))
    return results_by_layer
