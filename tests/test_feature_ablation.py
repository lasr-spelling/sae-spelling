from pytest import approx
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer

from sae_spelling.feature_ablation import calculate_individual_feature_ablations


def test_calculate_individual_feature_ablations_returns_0_if_feat_didnt_fire(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    output = calculate_individual_feature_ablations(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        sae=gpt2_l4_sae,
        ablate_features=[0, 10],
        ablate_token_index=10,
        batch_size=10,
    )
    assert output.ablation_scores[0] == approx(0.0, abs=1e-5)
    assert output.ablation_scores[10] == approx(0.0, abs=1e-5)


def test_calculate_individual_feature_ablations_has_non_zero_vals_for_firing_features(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    # ablate "John" feature: https://www.neuronpedia.org/gpt2-small/4-res-jb/1362
    # 1024 is another random feature that activates, but less strongly
    output = calculate_individual_feature_ablations(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        sae=gpt2_l4_sae,
        ablate_features=[1362, 1024, 10],
        ablate_token_index=-5,  # second John index
        batch_size=10,
    )
    # just ensure we've outputting something for the original metric
    assert abs(output.original_score) > 0.1
    assert isinstance(output.original_score, float)
    # 1362 should drop the metric by a lot more than 1024 should do anything
    assert output.ablation_scores[1362] < -0.9
    assert isinstance(output.ablation_scores[1362], float)
    assert abs(output.ablation_scores[1024]) < 0.2
    assert output.ablation_scores[10] == approx(0.0, abs=1e-5)


def test_calculate_individual_feature_ablations_ablates_all_firing_features_by_default(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    # ablate "John" feature: https://www.neuronpedia.org/gpt2-small/4-res-jb/1362
    # 1024 is another random feature that activates, but less strongly
    output = calculate_individual_feature_ablations(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        sae=gpt2_l4_sae,
        ablate_token_index=-5,  # second John index
        batch_size=10,
    )
    assert 1362 in output.ablation_scores
    assert 1024 in output.ablation_scores
    assert 10 not in output.ablation_scores
    firing_features = (
        output.sae_cache.feature_acts[0, -5, :].nonzero().squeeze().tolist()
    )
    assert output.ablation_scores.keys() == set(firing_features)


def test_calculate_individual_feature_ablations_ignores_features_firing_below_firing_threshold(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    # ablate "John" feature: https://www.neuronpedia.org/gpt2-small/4-res-jb/1362
    # 1024 is another random feature that activates, but less strongly
    output = calculate_individual_feature_ablations(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        sae=gpt2_l4_sae,
        ablate_token_index=-5,  # second John index
        batch_size=10,
        firing_threshold=3.5,
    )
    assert 1362 in output.ablation_scores
    assert 1024 not in output.ablation_scores


def test_calculate_individual_feature_ablations_gives_same_results_regardless_of_batch_size(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    assert gpt2_model.tokenizer is not None
    mary_token = gpt2_model.tokenizer.encode(" Mary")[0]

    def metric_fn(logits: Tensor) -> Tensor:
        return logits[:, -1, mary_token]

    output1 = calculate_individual_feature_ablations(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        sae=gpt2_l4_sae,
        ablate_features=[1362, 1024, 10],
        ablate_token_index=-5,  # second John index
        batch_size=10,
    )
    output2 = calculate_individual_feature_ablations(
        gpt2_model,
        "When John and Mary went to the shops, John gave the bag to",
        metric_fn=metric_fn,
        sae=gpt2_l4_sae,
        ablate_features=[1362, 1024, 10],
        ablate_token_index=-5,  # second John index
        batch_size=1,
    )
    assert output1.ablation_scores[1362] - output2.ablation_scores[1362] == approx(
        0.0, abs=1e-5
    )
    assert output1.ablation_scores[1024] - output2.ablation_scores[1024] == approx(
        0.0, abs=1e-5
    )
    assert output1.ablation_scores[10] - output2.ablation_scores[10] == approx(
        0.0, abs=1e-5
    )
