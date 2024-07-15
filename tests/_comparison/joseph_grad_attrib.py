import re
from typing import Callable

import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import ActivationCache, HookedTransformer

"""
Copied and modified from Joseph's notebook https://github.com/jbloomAus/SpellingSAEExperiment/blob/main/dev.ipynb
This is just for use in tests, to assert that our implementation gives identical results
"""

filter_resid_only = lambda name: "resid" in name


def get_cache_fwd_and_bwd(model, tokens, metric, filter=filter_resid_only):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter, backward_cache_hook, "bwd")

    logits = model(tokens)
    value = metric(logits)
    value.backward()
    model.reset_hooks()
    return (
        logits,
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


def get_logit_diff_metric(pos_token: str, neg_token: str, model: HookedTransformer):
    def logit_diff_metric(logits: torch.Tensor) -> torch.Tensor:
        positive_token_id = model.to_single_token(pos_token)
        negative_token_id = model.to_single_token(neg_token)
        pos_neg_logit_diff = (
            logits[0, -1, positive_token_id] - logits[0, -1, negative_token_id]  # type: ignore
        )
        return pos_neg_logit_diff

    return logit_diff_metric


def get_sae_out_all_layers(cache, sae_dict: dict[str, SAE]):
    sae_outs = {}
    feature_acts = {}
    for hook_point, sae in sae_dict.items():
        sae_feature_acts = sae.encode(cache[hook_point])
        sae_out = sae.decode(sae_feature_acts)
        sae_outs[hook_point] = sae_out.float()
        feature_acts[hook_point] = sae_feature_acts.float()

    return sae_outs, feature_acts


def gradient_based_attribution_all_layers(
    model: HookedTransformer,
    sparse_autoencoders: dict[str, SAE],
    prompt: str,
    metric: Callable[[torch.Tensor], torch.Tensor],
    position: int = 2,
):
    logits, clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
        model, prompt, metric
    )
    sae_outs, feature_acts_dict = get_sae_out_all_layers(
        clean_cache, sparse_autoencoders
    )

    attribution_dfs = []
    for hook_point, sparse_autoencoder in sparse_autoencoders.items():
        feature_acts = feature_acts_dict[hook_point]
        fired = (feature_acts[0, position, :] > 0).nonzero().squeeze()
        activations = feature_acts[0, position, :][fired]
        fired_directions = sparse_autoencoder.W_dec[fired]
        contributions = activations[:, None] * fired_directions
        logit_diff_grad = clean_grad_cache[hook_point][0, position].float()
        # attribution_scores = contributions @ pos_neg_logit_diff_direction
        attribution_scores = contributions @ logit_diff_grad

        attribution_df = pd.DataFrame(
            {
                "feature": fired.detach().cpu().numpy(),
                "activation": activations.detach().cpu().numpy(),
                "attribution": attribution_scores.detach().cpu().numpy(),
            }
        )
        attribution_df["layer"] = sparse_autoencoder.cfg.hook_name
        attribution_df["layer_idx"] = int(
            re.search(r"blocks.(\d+).hook_.*", sparse_autoencoder.cfg.hook_name).group(  # type: ignore
                1
            )
        ) + 1 * ("post" in sparse_autoencoder.cfg.hook_name)
        attribution_df["position"] = position

        attribution_dfs.append(attribution_df)

    attribution_df = pd.concat(attribution_dfs)
    attribution_df["feature"] = attribution_df.feature.astype(str)
    attribution_df["layer"] = attribution_df.layer.astype("category")

    tokens = model.to_str_tokens(prompt)
    unique_tokens = [f"{i}/{tokens[i]}" for i in range(len(tokens))]
    attribution_df["unique_token"] = attribution_df["position"].apply(
        lambda x: unique_tokens[x]
    )

    return attribution_df
