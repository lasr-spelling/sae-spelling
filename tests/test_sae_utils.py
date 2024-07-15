import pytest
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_spelling.sae_utils import apply_saes_and_run


def test_apply_saes_and_run_returns_identical_output_when_including_sae_error(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE, gpt2_l5_sae: SAE
):
    tokens = gpt2_model.to_tokens(["Hello, world!", "Ok then."])
    original_output, original_cache = gpt2_model.run_with_cache(
        tokens, names_filter=["blocks.4.mlp.hook_post", "blocks.5.mlp.hook_post"]
    )

    output = apply_saes_and_run(
        gpt2_model,
        {
            gpt2_l4_sae.cfg.hook_name: gpt2_l4_sae,
            gpt2_l5_sae.cfg.hook_name: gpt2_l5_sae,
        },
        input=tokens,
        include_error_term=True,
        track_model_hooks=["blocks.4.mlp.hook_post", "blocks.5.mlp.hook_post"],
    )
    with_sae_output = output.model_output
    with_sae_cache = output.model_activations
    sae_caches = output.sae_activations
    assert isinstance(original_output, torch.Tensor)
    assert torch.allclose(original_output, with_sae_output, atol=1e-4)
    assert original_cache.keys() == with_sae_cache.keys()
    for key, value in original_cache.items():
        assert torch.allclose(value, with_sae_cache[key], atol=1e-4)
    assert set(sae_caches.keys()) == {
        gpt2_l4_sae.cfg.hook_name,
        gpt2_l5_sae.cfg.hook_name,
    }
    for sae_cache in sae_caches.values():
        assert sae_cache.feature_acts.shape == (2, 5, 24576)
        assert sae_cache.sae_in.shape == (2, 5, 768)
        assert sae_cache.sae_out.shape == (2, 5, 768)
        assert sae_cache.sae_error.shape == (2, 5, 768)


@pytest.mark.parametrize("include_error_term", [False, True])
def test_apply_saes_works_with_fp16_models(
    gpt2_model: HookedTransformer,
    gpt2_l4_sae: SAE,
    gpt2_l5_sae: SAE,
    include_error_term: bool,
):
    tokens = gpt2_model.to_tokens(["Hello, world!", "Ok then."])
    output = apply_saes_and_run(
        gpt2_model.to(torch.float16),  # type: ignore
        {
            gpt2_l4_sae.cfg.hook_name: gpt2_l4_sae,
            gpt2_l5_sae.cfg.hook_name: gpt2_l5_sae,
        },
        input=tokens,
        include_error_term=include_error_term,
        track_model_hooks=["blocks.4.mlp.hook_post", "blocks.5.mlp.hook_post"],
    )
    sae_caches = output.sae_activations
    assert set(sae_caches.keys()) == {
        gpt2_l4_sae.cfg.hook_name,
        gpt2_l5_sae.cfg.hook_name,
    }
    for sae_cache in sae_caches.values():
        assert sae_cache.feature_acts.shape == (2, 5, 24576)
        assert sae_cache.sae_in.shape == (2, 5, 768)
        assert sae_cache.sae_out.shape == (2, 5, 768)
        assert sae_cache.sae_error.shape == (2, 5, 768)
