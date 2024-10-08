import torch
from nnsight import NNsight
from pytest import approx
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer

from sae_spelling.feature_attribution import (
    _get_interpolation_acts,
    calculate_feature_attribution,
    calculate_integrated_gradient_attribution_patching,
)
from tests._comparison.feature_circuits_attribution import pe_ig
from tests._comparison.joseph_grad_attrib import gradient_based_attribution_all_layers
from tests.test_util import set_sae_dtype


def test_calculate_feature_attribution_returns_values_that_look_reasonable(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

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
        return logits[:, -1, mary_token]

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
        return logits[:, -1, mary_token]

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
        return logits[:, -1, mary_token]

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


def test_calculate_integrated_gradient_attribution_patching_returns_values_that_look_reasonable(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    attrib = calculate_integrated_gradient_attribution_patching(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        track_hook_points=["blocks.3.mlp.hook_post"],
        include_saes={
            "blocks.3.hook_resid_pre": gpt2_l4_sae,
            "blocks.4.hook_resid_pre": gpt2_l4_sae,
        },
        patch_indices=-5,  # Final "John" index
        include_error_term=True,
        batch_size=10,
    )
    assert attrib.model_attributions.keys() == {"blocks.3.mlp.hook_post"}
    assert attrib.model_activations.keys() == {"blocks.3.mlp.hook_post"}
    assert attrib.sae_feature_attributions.keys() == {
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


def test_calculate_integrated_gradient_attribution_patching_zero_patching_all_indices_matches_the_sparse_feature_circuits_implementation(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    text = "When John and Mary went to the shops, John gave the bag to"
    toks = gpt2_model.to_tokens(text)

    attrib = calculate_integrated_gradient_attribution_patching(
        gpt2_model,
        text,
        metric_fn=metric_fn,
        include_saes={
            "blocks.4.hook_resid_post": gpt2_l4_sae,
        },
        include_error_term=True,
        batch_size=10,
    )

    nn_model = NNsight(gpt2_model)
    submodule = nn_model.blocks[4]
    dictionaries = {submodule: gpt2_l4_sae}  # type: ignore
    nn_metric = lambda model: metric_fn(model.unembed.output)
    saprmarks_attrib = pe_ig(
        clean=toks,
        patch=None,
        model=nn_model,
        submodules=[submodule],
        dictionaries=dictionaries,
        metric_fn=nn_metric,
        steps=10,
    )

    assert torch.allclose(
        attrib.sae_feature_activations["blocks.4.hook_resid_post"],
        -1 * saprmarks_attrib.deltas[submodule].act,
    )
    assert torch.allclose(
        attrib.sae_feature_grads["blocks.4.hook_resid_post"],
        saprmarks_attrib.grads[submodule].act,
    )
    assert torch.allclose(
        attrib.sae_error_grads["blocks.4.hook_resid_post"],
        saprmarks_attrib.grads[submodule].res,
    )
    assert torch.allclose(
        attrib.sae_feature_attributions["blocks.4.hook_resid_post"],
        saprmarks_attrib.effects[submodule].act,
    )


def test_calculate_integrated_gradient_attribution_patching_with_corrupted_input_matches_the_sparse_feature_circuits_implementation(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    text = "When John and Mary went to the shops, John gave the bag to"
    corrupted = "When John and Mary went to the shops, Mary gave the bag to"
    toks = gpt2_model.to_tokens(text)
    toks_corrupted = gpt2_model.to_tokens(corrupted)

    attrib = calculate_integrated_gradient_attribution_patching(
        gpt2_model,
        text,
        corrupted_input=corrupted,
        metric_fn=metric_fn,
        include_saes={
            "blocks.4.hook_resid_post": gpt2_l4_sae,
        },
        include_error_term=True,
        batch_size=10,
    )

    nn_model = NNsight(gpt2_model)
    submodule = nn_model.blocks[4]
    dictionaries = {submodule: gpt2_l4_sae}  # type: ignore
    nn_metric = lambda model: metric_fn(model.unembed.output)
    saprmarks_attrib = pe_ig(
        clean=toks,
        patch=toks_corrupted,
        model=nn_model,
        submodules=[submodule],
        dictionaries=dictionaries,
        metric_fn=nn_metric,
        steps=10,
    )

    assert torch.allclose(
        attrib.sae_feature_grads["blocks.4.hook_resid_post"],
        saprmarks_attrib.grads[submodule].act,
    )
    assert torch.allclose(
        attrib.sae_error_grads["blocks.4.hook_resid_post"],
        saprmarks_attrib.grads[submodule].res,
    )
    assert torch.allclose(
        attrib.sae_feature_attributions["blocks.4.hook_resid_post"],
        saprmarks_attrib.effects[submodule].act,
    )


def test_get_interpolation_acts_zero_ablates_if_no_corrupted_input_is_provided():
    clean_acts = {
        "act1": torch.tensor([1.0, 2.0]).repeat(1, 3, 1),
        "act2": torch.tensor([4.0, 6.0]).repeat(1, 3, 1),
    }
    interpolated_acts = _get_interpolation_acts(
        clean_acts, None, patch_indices=[(-2, -2)], interpolation_steps=3
    )
    assert len(interpolated_acts["act1"]) == 3
    assert len(interpolated_acts["act2"]) == 3
    for i in range(3):
        assert torch.all(interpolated_acts["act1"][i][0] == clean_acts["act1"][0])
        assert torch.all(interpolated_acts["act1"][i][-1] == clean_acts["act1"][-1])
        assert torch.all(interpolated_acts["act2"][i][0] == clean_acts["act2"][0])
        assert torch.all(interpolated_acts["act2"][i][-1] == clean_acts["act2"][-1])

    assert torch.all(interpolated_acts["act1"][0][-2] == clean_acts["act1"])
    assert torch.all(interpolated_acts["act1"][1][-2] == clean_acts["act1"] * 2 / 3)
    assert torch.all(interpolated_acts["act1"][2][-2] == clean_acts["act1"] / 3)


def test_get_interpolation_acts_works_with_multiple_indices():
    clean_acts = {"act": torch.tensor([1.0, 2.0]).repeat(1, 3, 1)}
    interpolated_acts = _get_interpolation_acts(
        clean_acts, None, patch_indices=[(-2, -2), (0, 0)], interpolation_steps=3
    )
    assert len(interpolated_acts["act"]) == 3
    for i in range(3):
        assert torch.all(interpolated_acts["act"][i][-1] == clean_acts["act"][-1])
    for idx in [0, -2]:
        assert torch.all(interpolated_acts["act"][0][idx] == clean_acts["act"])
        assert torch.all(interpolated_acts["act"][1][idx] == clean_acts["act"] * 2 / 3)
        assert torch.all(interpolated_acts["act"][2][idx] == clean_acts["act"] / 3)
