from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_spelling.sae_utils import (
    ApplySaesAndRunOutput,
    SaeReconstructionCache,
    apply_saes_and_run,
)
from sae_spelling.util import batchify, listify

EPS = 1e-8


@dataclass
class AttributionGrads:
    metric: torch.Tensor
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]


@dataclass
class Attribution:
    model_attributions: dict[str, torch.Tensor]
    model_activations: dict[str, torch.Tensor]
    model_grads: dict[str, torch.Tensor]
    sae_feature_attributions: dict[str, torch.Tensor]
    sae_feature_activations: dict[str, torch.Tensor]
    sae_feature_grads: dict[str, torch.Tensor]
    sae_error_grads: dict[str, torch.Tensor]


@dataclass
class PatchedAttribution:
    model_attributions: dict[str, torch.Tensor]
    model_activations: dict[str, torch.Tensor]
    model_grads: dict[str, torch.Tensor]
    sae_feature_attributions: dict[str, torch.Tensor]
    sae_feature_activations: dict[str, torch.Tensor]
    sae_feature_grads: dict[str, torch.Tensor]
    sae_error_grads: dict[str, torch.Tensor]
    corrupted_model_activations: dict[str, torch.Tensor] | None
    corrupted_sae_feature_activations: dict[str, torch.Tensor] | None


def calculate_attribution_grads(
    model: HookedTransformer,
    prompt: str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> AttributionGrads:
    """
    Wrapper around apply_saes_and_run that calculates gradients wrt to the metric_fn.
    Tracks grads for both SAE feature and model neurons, and returns them in a structured format.
    """
    output = apply_saes_and_run(
        model,
        saes=include_saes or {},
        input=prompt,
        return_type="logits" if return_logits else "loss",
        track_model_hooks=track_hook_points,
        include_error_term=include_error_term,
        track_grads=True,
    )
    metric = metric_fn(output.model_output)
    output.zero_grad()
    metric.backward()
    return AttributionGrads(
        metric=metric,
        model_output=output.model_output,
        model_activations=output.model_activations,
        sae_activations=output.sae_activations,
    )


def calculate_feature_attribution(
    model: HookedTransformer,
    input: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> Attribution:
    """
    Calculate feature attribution for SAE features and model neurons following
    the procedure in https://transformer-circuits.pub/2024/march-update/index.html#feature-heads.
    This include the SAE error term by default, so inserting the SAE into the calculation is
    guaranteed to not affect the model output. This can be disabled by setting `include_error_term=False`.

    Args:
        model: The model to calculate feature attribution for.
        input: The input to the model.
        metric_fn: A function that takes the model output and returns a scalar metric.
        track_hook_points: A list of model hook points to track activations for, if desired
        include_saes: A dictionary of SAEs to include in the calculation. The key is the hook point to apply the SAE to.
        return_logits: Whether to return the model logits or loss. This is passed to TLens, so should match whatever the metric_fn expects (probably logits)
        include_error_term: Whether to include the SAE error term in the calculation. This is recommended, as it ensures that the SAE will not affecting the model output.
    """
    # first, calculate gradients wrt to the metric_fn.
    # these will be multiplied with the activation values to get the attributions
    outputs_with_grads = calculate_attribution_grads(
        model,
        input,
        metric_fn,
        track_hook_points,
        include_saes=include_saes,
        return_logits=return_logits,
        include_error_term=include_error_term,
    )
    model_attributions = {}
    model_activations = {}
    model_grads = {}
    sae_feature_attributions = {}
    sae_feature_activations = {}
    sae_feature_grads = {}
    sae_error_grads = {}
    # this code is long, but all it's doing is multiplying the grads by the activations
    # and recording grads, acts, and attributions in dictionaries to return to the user
    with torch.no_grad():
        for name, act in outputs_with_grads.model_activations.items():
            assert act.grad is not None
            raw_activation = act.detach().clone()
            model_attributions[name] = (act.grad * raw_activation).detach().clone()
            model_activations[name] = raw_activation
            model_grads[name] = act.grad.detach().clone()
        for name, act in outputs_with_grads.sae_activations.items():
            assert act.feature_acts.grad is not None
            assert act.sae_out.grad is not None
            raw_activation = act.feature_acts.detach().clone()
            sae_feature_attributions[name] = (
                (act.feature_acts.grad * raw_activation).detach().clone()
            )
            sae_feature_activations[name] = raw_activation
            sae_feature_grads[name] = act.feature_acts.grad.detach().clone()
            if include_error_term:
                assert act.sae_error.grad is not None
                sae_error_grads[name] = act.sae_error.grad.detach().clone()
        return Attribution(
            model_attributions=model_attributions,
            model_activations=model_activations,
            model_grads=model_grads,
            sae_feature_attributions=sae_feature_attributions,
            sae_feature_activations=sae_feature_activations,
            sae_feature_grads=sae_feature_grads,
            sae_error_grads=sae_error_grads,
        )


def calculate_integrated_gradient_attribution_patching(
    model: HookedTransformer,
    input: str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    patch_indices: int | list[int | tuple[int, int]],
    corrupted_input: str | None = None,
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
    interpolation_steps: int = 10,
    batch_size: int = 1,
) -> PatchedAttribution:
    """
    Calculate attribution scores for SAE features and model neurons based on the integrated gradients method.
    This procedure comes from the [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) paper. This
    should be a more faithful approximation than pure attirbution while running slower (but still faster than ablation).
    This include the SAE error term by default, so inserting the SAE into the calculation is
    guaranteed to not affect the model output. This can be disabled by setting `include_error_term=False`.

    Args:
        model: The model to calculate feature attribution for.
        input: The input to the model.
        metric_fn: A function that takes the model output and returns a scalar metric.
        patch_indices: The indices to patch in the input. If a single int, will patch that token index with the corresponding
            index from the corrupted input. If a list of ints, will patch each index with the corresponding index in the corrupted input.
            If a list of tuples, the first index is the index to patch, and the second index is the index in the corrupted input to extract the patch from.
        corrupted_input: The corrupted input to extract patches from. If None, will use zeros (ablation patching).
        track_hook_points: A list of model hook points to track activations for, if desired
        include_saes: A dictionary of SAEs to include in the calculation. The key is the hook point to apply the SAE to.
        return_logits: Whether to return the model logits or loss. This is passed to TLens, so should match whatever the metric_fn expects (probably logits)
        include_error_term: Whether to include the SAE error term in the calculation. This is recommended, as it ensures that the SAE will not affecting the model output.
        interpolation_steps: The number of interpolation steps to use for the integrated gradients calculation.
    """
    with torch.no_grad():
        input_toks = model.to_tokens(input)
        original_output = apply_saes_and_run(
            model,
            saes=include_saes or {},
            input=input_toks,
            include_error_term=include_error_term,
            track_model_hooks=track_hook_points,
            return_type="logits" if return_logits else "loss",
        )
        original_sae_acts = _extract_sae_acts(original_output)
        corrupted_output = None
        if corrupted_input is not None:
            corrupted_output = apply_saes_and_run(
                model,
                saes=include_saes or {},
                input=corrupted_input,
                include_error_term=include_error_term,
                track_model_hooks=track_hook_points,
                return_type="logits" if return_logits else "loss",
            )
        corrupted_sae_acts = (
            _extract_sae_acts(corrupted_output) if corrupted_output else None
        )
        patch_index_tuples: list[tuple[int, int]] = [
            (i, i) if isinstance(i, int) else i for i in listify(patch_indices)
        ]
        interpolated_hook_point_acts = _get_interpolation_acts(
            original_output.model_activations,
            corrupted_output.model_activations if corrupted_output else None,
            patch_index_tuples,
            interpolation_steps,
        )
        interpolated_sae_acts = _get_interpolation_acts(
            original_sae_acts,
            corrupted_sae_acts,
            patch_index_tuples,
            interpolation_steps,
        )
        model_patches: list[tuple[Literal["model", "sae"], str, torch.Tensor]] = [
            ("model", hook_point, act)
            for hook_point, acts in interpolated_hook_point_acts.items()
            for act in acts
        ]
        sae_patches: list[tuple[Literal["model", "sae"], str, torch.Tensor]] = [
            ("sae", hook_point, act)
            for hook_point, acts in interpolated_sae_acts.items()
            for act in acts
        ]
        sae_grads: dict[str, list[torch.Tensor]] = defaultdict(list)
        sae_error_grads: dict[str, list[torch.Tensor]] = defaultdict(list)
        model_grads: dict[str, list[torch.Tensor]] = defaultdict(list)
        patches_to_run = model_patches + sae_patches
    for batch in batchify(patches_to_run, batch_size=batch_size):
        with torch.no_grad():
            batch_inputs = input_toks.repeat(len(batch), 1)
        with ExitStack() as stack:
            for i, (hook_type, hook_point, act) in enumerate(batch):
                hook = patch_hook(i, act)
                if hook_type == "model":
                    stack.enter_context(model.hooks(fwd_hooks=[(hook_point, hook)]))
                else:
                    assert include_saes is not None
                    sae = include_saes[hook_point]
                    stack.enter_context(
                        sae.hooks(fwd_hooks=[("hook_sae_acts_post", hook)])
                    )
            attribution = calculate_feature_attribution(
                model,
                input=batch_inputs,
                include_saes=include_saes,
                metric_fn=metric_fn,
                include_error_term=include_error_term,
                track_hook_points=track_hook_points,
                return_logits=return_logits,
            )
            with torch.no_grad():
                for i, (hook_type, hook_point, act) in enumerate(batch):
                    if hook_type == "model":
                        model_grads[hook_point].append(
                            attribution.model_grads[hook_point][i]
                        )
                    else:
                        sae_grads[hook_point].append(
                            attribution.sae_feature_grads[hook_point][i]
                        )
                        if include_error_term:
                            sae_error_grads[hook_point].append(
                                attribution.sae_error_grads[hook_point][i]
                            )
    with torch.no_grad():
        mean_sae_grads = {
            hook_point: torch.stack(grads).mean(dim=0).detach().clone()
            for hook_point, grads in sae_grads.items()
        }
        mean_model_grads = {
            hook_point: torch.stack(grads).mean(dim=0).detach().clone()
            for hook_point, grads in model_grads.items()
        }
        mean_sae_error_grads = {
            hook_point: torch.stack(grads).mean(dim=0).detach().clone()
            for hook_point, grads in sae_error_grads.items()
        }
        model_attributions = {}
        sae_attributions = {}
        for hook_point, grads in mean_model_grads.items():
            delta = -original_output.model_activations[hook_point]
            if corrupted_output:
                delta += corrupted_output.model_activations[hook_point]
            model_attributions[hook_point] = (grads * delta).detach().clone()
        for hook_point, grads in mean_sae_grads.items():
            delta = -original_sae_acts[hook_point]
            if corrupted_sae_acts:
                delta += corrupted_sae_acts[hook_point]
            sae_attributions[hook_point] = (grads * delta).detach().clone()

        return PatchedAttribution(
            model_attributions=model_attributions,
            model_activations=original_output.model_activations,
            model_grads=mean_model_grads,
            sae_feature_attributions=sae_attributions,
            sae_feature_activations=original_sae_acts,
            sae_feature_grads=mean_sae_grads,
            sae_error_grads=mean_sae_error_grads,
            corrupted_model_activations=(
                corrupted_output.model_activations if corrupted_output else None
            ),
            corrupted_sae_feature_activations=corrupted_sae_acts,
        )


def patch_hook(
    batch_idx: int,
    patch_act: torch.Tensor,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor | None]:
    def patch_hook_fn(input: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        # doing this masking feels inefficient, but otherwise torch whines about in-place operations
        mask = torch.ones_like(input)
        mask[batch_idx] = 0
        return input * mask + patch_act * (1 - mask)

    return patch_hook_fn


def _extract_sae_acts(
    output: ApplySaesAndRunOutput,
) -> dict[str, torch.Tensor]:
    return {name: cache.feature_acts for name, cache in output.sae_activations.items()}


@torch.no_grad()
def _get_interpolation_acts(
    clean_acts: dict[str, torch.Tensor],
    corrupted_acts: dict[str, torch.Tensor] | None,
    patch_indices: list[tuple[int, int]],
    interpolation_steps: int,
) -> dict[str, list[torch.Tensor]]:
    interpolated_acts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, act in clean_acts.items():
        for clean_index, corrupted_index in patch_indices:
            clean_val = act[0, clean_index, :]
            base_corrupted_val = (
                corrupted_acts[name][0, corrupted_index, :]
                if corrupted_acts is not None
                else torch.zeros_like(clean_val)
            )
            for i in range(interpolation_steps):
                alpha = i / interpolation_steps
                patch_act = act[0].clone()
                patch_act[clean_index, :] = (
                    alpha * clean_val + (1 - alpha) * base_corrupted_val
                )
                interpolated_acts[name].append(patch_act)
    return interpolated_acts
