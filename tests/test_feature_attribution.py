import torch
from pytest import approx
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer

from sae_spelling.feature_attribution import calculate_feature_attribution
from tests._comparison.joseph_grad_attrib import gradient_based_attribution_all_layers
from tests.test_util import set_sae_dtype


def test_calculate_feature_attribution_returns_values_that_look_reasonable(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[-1, -1, mary_token]

    attrib = calculate_feature_attribution(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        track_hook_points=["blocks.3.mlp.hook_post"],
        include_saes={
            "blocks.3.hook_resid_pre": gpt2_l4_sae,
            "blocks.4.hook_resid_pre": gpt2_l4_sae,
        },
        include_error_term=True,
    )
    assert attrib.model_attributions.keys() == {"blocks.3.mlp.hook_post"}
    assert attrib.model_activations.keys() == {"blocks.3.mlp.hook_post"}
    assert attrib.sae_feature_attributions.keys() == {
        "blocks.3.hook_resid_pre",
        "blocks.4.hook_resid_pre",
    }
    assert attrib.sae_errors_attribution_proportion.keys() == {
        "blocks.3.hook_resid_pre",
        "blocks.4.hook_resid_pre",
    }
    assert attrib.sae_feature_activations.keys() == {
        "blocks.3.hook_resid_pre",
        "blocks.4.hook_resid_pre",
    }
    for value in attrib.sae_feature_attributions.values():
        assert value.shape == (1, 15, 24576)
        # just to assert that at least some features / gradients are non-zero
        assert value.abs().sum() > 0.05
    for value in attrib.sae_feature_activations.values():
        assert value.shape == (1, 15, 24576)
        # these should all be positive
        assert value.sum() > 100


def test_calculate_feature_attribution_results_match_josephs_implementation(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE, gpt2_l5_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]
    prompt = "When John and Mary went to the shops, John gave the bag to"
    saes_dict = {
        "blocks.4.hook_resid_pre": gpt2_l4_sae,
        "blocks.5.hook_resid_pre": gpt2_l5_sae,
    }

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[-1, -1, mary_token]

    attrib = calculate_feature_attribution(
        gpt2_model,
        prompt,
        metric_fn=metric_fn,
        include_saes=saes_dict,
    )

    joseph_attrib = gradient_based_attribution_all_layers(
        gpt2_model,
        saes_dict,
        prompt,
        metric_fn,
        position=2,
    )

    for row in joseph_attrib.itertuples():
        # pandas doesn't play nicely with type checking
        hook_point: str = row.layer  # type: ignore
        feature: int = int(row.feature)  # type: ignore
        attribution: float = row.attribution  # type: ignore
        activation: float = row.activation  # type: ignore
        assert attrib.sae_feature_attributions[hook_point][0, 2, feature] == approx(
            attribution,
            abs=1e-5,
        )
        assert attrib.sae_feature_activations[hook_point][0, 2, feature] == approx(
            activation,
            abs=1e-5,
        )


def test_calculate_feature_attribution_works_with_fp16_models(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[-1, -1, mary_token]

    attrib = calculate_feature_attribution(
        gpt2_model.to(torch.float16),  # type: ignore
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        track_hook_points=["blocks.3.mlp.hook_post"],
        include_saes={
            "blocks.3.hook_resid_pre": set_sae_dtype(gpt2_l4_sae, torch.float16),
            "blocks.4.hook_resid_pre": set_sae_dtype(gpt2_l4_sae, torch.float16),
        },
        include_error_term=True,
    )
    for value in attrib.sae_feature_attributions.values():
        assert value.shape == (1, 15, 24576)
        # just to assert that at least some features / gradients are non-zero
        assert value.abs().sum() > 0.05
    for value in attrib.sae_feature_activations.values():
        assert value.shape == (1, 15, 24576)
        # these should all be positive
        assert value.sum() > 100


def test_calculate_feature_attribution_returns_identical_model_attribution_with_and_without_saes(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[-1, -1, mary_token]

    prompt = "When John and Mary went to the shops, John gave the bag to"
    track_hook_points = [
        "blocks.3.mlp.hook_post",
        "blocks.4.mlp.hook_post",
        "blocks.5.mlp.hook_post",
        "blocks.2.attn.hook_z",
        "blocks.3.attn.hook_z",
        "blocks.2.hook_attn_in",
    ]

    without_saes_attrib = calculate_feature_attribution(
        gpt2_model,
        prompt,
        metric_fn=metric_fn,
        track_hook_points=track_hook_points,
    )

    with_saes_attrib = calculate_feature_attribution(
        gpt2_model,
        prompt,
        metric_fn=metric_fn,
        track_hook_points=track_hook_points,
        include_saes={
            "blocks.3.hook_resid_pre": gpt2_l4_sae,
            "blocks.4.hook_resid_pre": gpt2_l4_sae,
        },
        include_error_term=True,
    )

    for with_sae_val, without_sae_val in zip(
        with_saes_attrib.model_attributions.values(),
        without_saes_attrib.model_attributions.values(),
    ):
        assert torch.allclose(with_sae_val, without_sae_val, atol=1e-5)
        assert with_sae_val.abs().sum() > 0.05
