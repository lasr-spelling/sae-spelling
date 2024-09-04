from unittest.mock import MagicMock

import pytest
import torch
from sae_lens import SAE
from torch.nn.functional import normalize
from transformer_lens import HookedTransformer

from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator


def test_FeatureAbsorptionCalculator_filter_prompts_removes_prompts_where_main_features_fired(
    gpt2_model: HookedTransformer,
):
    words = [" cat", " dog", " fish"]
    calculator = FeatureAbsorptionCalculator(gpt2_model, icl_word_list=["dog"])
    prompts = calculator._build_prompts(words)
    mock_sae = MagicMock()
    sae_acts = torch.zeros(3, 10, 768)
    sae_acts[0, :, 3] = 1.0  # feature 3 fires on prompt 0 at all token positions
    mock_sae.encode.return_value = sae_acts
    mock_sae.cfg.hook_name = "blocks.0.hook_resid_post"

    filtered_prompts = calculator._filter_prompts(prompts, mock_sae, [3, 4])

    assert len(filtered_prompts) == 2
    assert filtered_prompts == prompts[1:]


def test_FeatureAbsorptionCalculator_calculate_absorption_sampled_results_look_reasonable(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    words = [" cat", " chair", " car"]
    calculator = FeatureAbsorptionCalculator(
        gpt2_model, icl_word_list=["dog"], topk_feats=10
    )
    probe_dir = normalize(torch.randn(768), dim=-1)
    c_token = gpt2_model.tokenizer.encode(" C")[0]  # type: ignore

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, c_token]

    sampled_results = calculator.calculate_absorption_sampled(
        gpt2_l4_sae,
        words,
        probe_dir,
        metric_fn=metric_fn,
        main_feature_ids=[1, 2, 3],
        filter_prompts=False,
    )
    with torch.no_grad():
        assert sampled_results.sample_portion == 1.0
        assert sampled_results.main_feature_ids == [1, 2, 3]
        assert len(sampled_results.sample_results) == 3
        for sample in sampled_results.sample_results:
            assert sample.word in words
            assert len(sample.main_feature_scores) == 3
            assert len(sample.top_ablation_feature_scores) == 10
            for feat_score in sample.main_feature_scores:
                assert feat_score.feature_id in [1, 2, 3]
                sae_dir = normalize(gpt2_l4_sae.W_dec[feat_score.feature_id], dim=-1)
                assert feat_score.probe_cos_sim == pytest.approx(
                    (probe_dir @ sae_dir).item(), abs=1e-5
                )
            for feat_score in sample.top_ablation_feature_scores:
                sae_dir = normalize(gpt2_l4_sae.W_dec[feat_score.feature_id], dim=-1)
                assert feat_score.probe_cos_sim == pytest.approx(
                    (probe_dir @ sae_dir).item(), abs=1e-5
                )
