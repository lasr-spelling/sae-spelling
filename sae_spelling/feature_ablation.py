from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_spelling.sae_utils import (
    SaeReconstructionCache,
    apply_saes_and_run,
)
from sae_spelling.util import batchify


@dataclass
class FeatureAblationsOutput:
    sae_cache: SaeReconstructionCache
    ablation_scores: dict[int, float]
    original_score: float


@torch.no_grad()
def calculate_individual_feature_ablations(
    model: HookedTransformer,
    input: str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    sae: SAE,
    ablate_token_index: int,  # TODO: this can be extended to multiple tokens
    ablate_features: Sequence[int] | None = None,
    return_logits: bool = True,
    batch_size: int = 10,
    show_progress: bool = True,
    # TODO: support not including the error term
) -> FeatureAblationsOutput:
    """
    Calculate the effect of ablating each feature individually. This means return the change in metric if the
    SAE feature never fired at all during the forward pass on the input.
    This is what feature attribution is approximating, but the signs are flipped since attribution asks "How important is a feature?",
    while ablation asks "what is the metric delta if the feature is deleted?".
    If we're unsure about how accurate attribution is, we can compare it to this function (after flipping the sign). This function is much slower, however.

    Args:
        model: The model to run the forward pass on.
        input: The input to the model.
        metric_fn: A function that takes the model output and returns a scalar metric.
        sae: The SAE to use for ablation.
        ablate_features: The indices of the features to ablate. If empty, ablate all firing features.
        ablate_token_index: The index of the token to ablate.
        return_logits: Whether to return the logits or the loss.
        batch_size: The batch size to use for ablation.
        show_progress: Whether to show a progress bar.
    """
    input_toks = model.to_tokens(input)
    hook_point = sae.cfg.hook_name
    original_output = apply_saes_and_run(
        model,
        saes={hook_point: sae},
        input=input_toks,
        return_type="logits" if return_logits else "loss",
        include_error_term=True,
    )
    original_score = metric_fn(original_output.model_output).item()
    if ablate_features is None:
        sae_acts = original_output.sae_activations[hook_point]
        ablate_features = (
            torch.nonzero(sae_acts.feature_acts[0, ablate_token_index])
            .squeeze(-1)
            .tolist()
        )
    ablation_scores = {}
    for batch in batchify(ablate_features, batch_size, show_progress=show_progress):
        feature_vals = original_output.sae_activations[hook_point].feature_acts[
            0, ablate_token_index, batch
        ]
        batch_deltas = -1 * feature_vals.unsqueeze(-1) * sae.W_dec[batch]

        def ablation_hook(value: torch.Tensor, hook: HookPoint):  # noqa: ARG001
            value[:, ablate_token_index, :] += batch_deltas
            return value

        outputs = model.run_with_hooks(
            input_toks.repeat(len(batch), 1),
            fwd_hooks=[(hook_point, ablation_hook)],
        )
        for feat_idx, output in zip(batch, outputs):
            score = metric_fn(output.unsqueeze(0)).item()
            ablation_scores[feat_idx] = score - original_score
    return FeatureAblationsOutput(
        sae_cache=original_output.sae_activations[hook_point],
        ablation_scores=ablation_scores,
        original_score=original_score,
    )
