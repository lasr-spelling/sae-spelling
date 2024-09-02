import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from tueplots import axes, bundles

from sae_spelling.experiments.common import (
    EXPERIMENTS_DIR,
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
    get_sparse_probing_raw_results_filename,
)
from sae_spelling.feature_attribution import (
    calculate_integrated_gradient_attribution_patching,
)
from sae_spelling.probing import LinearProbe
from sae_spelling.prompting import (
    SpellingPrompt,
    create_icl_prompt,
    first_letter_formatter,
)
from sae_spelling.util import batchify
from sae_spelling.vocab import LETTERS, LETTERS_UPPER, get_alpha_tokens

FEATURE_ABSORPTION_EXPERIMENT_NAME = "feature_absorption"

# the cosine similarity between the top ablation feature and the probe must be at least this high
ABSORPTION_PROBE_COS_THRESHOLD = 0.025

# the top ablation score must be at least this much larger than the second highest score
ABSORPTION_FEATURE_DELTA_THRESHOLD = 1.0


TEMPLATE = "{word} has the first letter:"
TOKEN_POS = -6


def ig_ablate(model: HookedTransformer, sae: SAE, prompt: SpellingPrompt):
    def letter_delta_metric(pos_letter: str):
        neg_letters = [
            f" {letter}" for letter in LETTERS_UPPER if pos_letter[-1].upper() != letter
        ]
        pos_letter_tok = model.tokenizer.encode(pos_letter)[-1]  # type: ignore
        neg_letter_toks = torch.tensor(
            [model.tokenizer.encode(neg_letter)[-1] for neg_letter in neg_letters]  # type: ignore
        )

        def metric_fn(logits):
            pos_logit = logits[:, -1, pos_letter_tok]
            neg_logits = logits[:, -1, neg_letter_toks]
            result = pos_logit - (neg_logits.sum(dim=-1) / len(neg_letters))
            return result

        return metric_fn

    attrib = calculate_integrated_gradient_attribution_patching(
        model=model,
        input=prompt.base,
        include_saes={sae.cfg.hook_name: sae},
        metric_fn=letter_delta_metric(pos_letter=prompt.answer),
        patch_indices=TOKEN_POS,
        batch_size=6,
        interpolation_steps=6,
    )
    with torch.no_grad():
        sae_acts = (
            attrib.sae_feature_activations[sae.cfg.hook_name][0, TOKEN_POS]
            .float()
            .cpu()
            .detach()
            .clone()
        )
        abl_scores = (
            attrib.sae_feature_attributions[sae.cfg.hook_name][0, TOKEN_POS]
            .float()
            .cpu()
            .detach()
            .clone()
        )
        return abl_scores, sae_acts


def enhance_df(df):
    df["delta"] = df["top_ablation_score"].abs() - df["second_ablation_score"].abs()
    df["is_absorption"] = (
        (df["top_ablation_feat_probe_cos"] > ABSORPTION_PROBE_COS_THRESHOLD)
        & (df["top_ablation_score"] < 0)
        & (df["delta"] > ABSORPTION_FEATURE_DELTA_THRESHOLD)
    )


@torch.inference_mode()
def prefilter_prompts(
    model: HookedTransformer,
    sae: SAE,
    vocab: list[str],
    tokens: list[str],
    split_feats: list[int],
    batch_size: int = 40,
) -> list[SpellingPrompt]:
    filtered_prompts: list[SpellingPrompt] = []
    for batch_toks in batchify(tokens, batch_size=batch_size):
        batch_prompts = [
            create_icl_prompt(
                tok,
                base_template=TEMPLATE,
                examples=vocab,
                answer_formatter=first_letter_formatter(capitalize=True),
                max_icl_examples=10,
            )
            for tok in batch_toks
        ]
        sae_in = model.run_with_cache([p.base for p in batch_prompts])[1][
            sae.cfg.hook_name
        ]
        sae_acts = sae.encode(sae_in)
        split_feats_active = (
            sae_acts[:, TOKEN_POS, split_feats].sum(dim=-1).float().tolist()
        )
        for tok, prompt, res in zip(batch_toks, batch_prompts, split_feats_active):
            if res < 1e-8:
                filtered_prompts.append(prompt)
    return filtered_prompts


def filter_and_gen_prompts(
    model: HookedTransformer,
    sae: SAE,
    likely_negs: dict[str, tuple[int, list[int], list[str]]],
) -> dict[str, tuple[int, list[int], list[SpellingPrompt]]]:
    results: dict[str, tuple[int, list[int], list[SpellingPrompt]]] = {}
    vocab = get_alpha_tokens(model.tokenizer)  # type: ignore
    for letter, (num_true_positives, split_feats, tokens) in tqdm(
        likely_negs.items(), desc="prefiltering prompts"
    ):
        prompts = prefilter_prompts(
            model, sae, vocab=vocab, tokens=tokens, split_feats=split_feats
        )
        random.shuffle(prompts)
        results[letter] = (num_true_positives, split_feats, prompts)
    return results


def calculate_ig_ablation_and_cos_sims(
    model: HookedTransformer,
    sae: SAE,
    probe: LinearProbe,
    likely_negs: dict[str, tuple[int, list[int], list[str]]],
    max_prompts_per_letter: int = 50,
) -> pd.DataFrame:
    results = []
    processed_prompts_and_toks = filter_and_gen_prompts(model, sae, likely_negs)
    total_items = sum(
        {
            letter: min(len(v[2]), max_prompts_per_letter)
            for letter, v in processed_prompts_and_toks.items()
        }.values()
    )
    if total_items == 0:
        return pd.DataFrame()
    with tqdm(total=total_items) as pbar:
        for letter, (
            num_true_positives,
            split_feats,
            prompts,
        ) in processed_prompts_and_toks.items():
            with torch.no_grad():
                cos_sims = (
                    torch.nn.functional.cosine_similarity(
                        probe.weights[LETTERS.index(letter)].unsqueeze(0).cuda(),
                        sae.W_dec,
                        dim=-1,
                    )
                    .float()
                    .cpu()
                )
            for prompt in prompts[:max_prompts_per_letter]:
                ig_scores, sae_acts = ig_ablate(model, sae, prompt)
                with torch.no_grad():
                    top_ig_feats = ig_scores.abs().topk(k=10).indices.tolist()
                    top_ig_scores = ig_scores[top_ig_feats].tolist()
                    feat_cos_sims = cos_sims[top_ig_feats].tolist()
                    result = {
                        "letter": letter,
                        "token": prompt.word,
                        "prompt": prompt.base,
                        "sample_portion": 1.0
                        if len(prompts) <= max_prompts_per_letter
                        else max_prompts_per_letter / len(prompts),
                        "num_true_positives": num_true_positives,
                        "split_feats": split_feats,
                        "split_feat_acts": sae_acts[split_feats].tolist(),
                        "top_ablation_feat": top_ig_feats[0],
                        "top_ablation_score": top_ig_scores[0],
                        "top_ablation_feat_probe_cos": feat_cos_sims[0],
                        "second_ablation_feat": top_ig_feats[1],
                        "second_ablation_score": top_ig_scores[1],
                        "second_ablation_feat_probe_cos": feat_cos_sims[1],
                        "ablation_scores": top_ig_scores,
                        "ablation_feats": top_ig_feats,
                        "ablation_feat_acts": sae_acts[top_ig_feats].tolist(),
                        "ablation_feat_prob_cos": feat_cos_sims,
                    }
                    results.append(result)
                    pbar.update(1)
    result_df = pd.DataFrame(results)
    enhance_df(result_df)
    return result_df


def get_likely_false_negative_tokens(
    auroc_f1_df: pd.DataFrame,
    sae_info: SaeInfo,
    sparse_probing_task_output_dir: Path,
    sparse_probing_sae_post_act: bool,
) -> dict[str, tuple[int, list[int], list[str]]]:
    results: dict[str, tuple[int, list[int], list[str]]] = {}
    raw_df = load_experiment_df(
        SPARSE_PROBING_EXPERIMENT_NAME,
        sparse_probing_task_output_dir
        / get_sparse_probing_raw_results_filename(
            sae_info, sparse_probing_sae_post_act
        ),
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
        num_true_positives = raw_df[
            (raw_df["answer_letter"] == letter)
            & (raw_df[f"score_probe_{letter}"] > 0)
            & (raw_df[f"score_sparse_sae_{letter}_k_{k}"] > 0)
        ].shape[0]
        results[letter] = (num_true_positives, split_feats, potential_false_negatives)
    return results


def load_and_run_calculate_ig_ablation_and_cos_sims(
    model: HookedTransformer,
    auroc_f1_df: pd.DataFrame,
    sae_info: SaeInfo,
    sparse_probing_task_output_dir: Path,
    sparse_probing_sae_post_act: bool,
) -> pd.DataFrame:
    layer = sae_info.layer
    probe = load_probe(layer=layer)
    sae = load_gemmascope_sae(layer, width=sae_info.width, l0=sae_info.l0)
    likely_negs = get_likely_false_negative_tokens(
        auroc_f1_df,
        sae_info,
        sparse_probing_task_output_dir,
        sparse_probing_sae_post_act,
    )
    return calculate_ig_ablation_and_cos_sims(model, sae, probe, likely_negs)


def _aggregate_results_df(
    results: dict[int, list[tuple[pd.DataFrame, SaeInfo]]],
) -> pd.DataFrame:
    combined = []
    for layer, result in results.items():
        for df, sae_info in result:
            if "sample_portion" not in df.columns:
                df["sample_portion"] = 1.0
            agg_df = (
                df[["letter", "is_absorption"]]
                .groupby(["letter"])
                .sum()
                .reset_index()
                .merge(
                    df[["letter", "sample_portion", "num_true_positives"]]
                    .groupby(["letter"])
                    .mean()
                    .reset_index()
                )
            )
            agg_df["num_absorption"] = (
                agg_df["is_absorption"] / agg_df["sample_portion"]
            )
            agg_df["absorption_rate"] = agg_df["num_absorption"] / (
                agg_df["num_absorption"] + agg_df["num_true_positives"]
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
    task: str = "first_letter",
    sparse_probing_sae_post_act: bool = True,
    force: bool = False,
    skip_1m_saes: bool = False,
) -> dict[int, list[tuple[pd.DataFrame, SaeInfo]]]:
    """
    NOTE: this experiments requires the results of the k-sparse probing experiments. Make sure to run them first.
    """
    task_output_dir = get_task_dir(experiment_dir, task=task)
    sparse_probing_task_output_dir = get_task_dir(
        sparse_probing_experiment_dir, task=task
    )
    model = load_gemma2_model()
    results_by_layer: dict[int, list[tuple[pd.DataFrame, SaeInfo]]] = defaultdict(list)
    with tqdm(total=len(layers)) as pbar:
        for layer in layers:
            pbar.set_description(f"Layer {layer}")
            sae_infos = get_gemmascope_saes_info(layer)
            for sae_info in sae_infos:
                if skip_1m_saes and sae_info.width == 1_000_000:
                    continue
                auroc_f1_df = load_experiment_df(
                    SPARSE_PROBING_EXPERIMENT_NAME,
                    sparse_probing_task_output_dir
                    / get_sparse_probing_raw_results_filename(
                        sae_info, sparse_probing_sae_post_act
                    ),
                )
                df_path = (
                    task_output_dir
                    / f"layer_{layer}_width_{sae_info.width}_l0_{sae_info.l0}.parquet"
                )
                df = load_df_or_run(
                    lambda: load_and_run_calculate_ig_ablation_and_cos_sims(
                        model,
                        auroc_f1_df,
                        sae_info,
                        sparse_probing_task_output_dir,
                        sparse_probing_sae_post_act,
                    ),
                    df_path,
                    force=force,
                )
                results_by_layer[layer].append((df, sae_info))
    return results_by_layer
