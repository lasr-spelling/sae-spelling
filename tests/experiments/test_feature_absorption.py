from sae_lens import SAE
from syrupy.assertion import SnapshotAssertion
from transformer_lens import HookedTransformer

from sae_spelling.experiments.feature_absorption import (
    calculate_ig_ablation_and_cos_sims,
)
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
from sae_spelling.probing import LinearProbe
from sae_spelling.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
)


def test_calculate_ig_ablation_and_cos_sims_gives_sane_results(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE, snapshot: SnapshotAssertion
):
    fake_probe = LinearProbe(768, 26)
    calculator = FeatureAbsorptionCalculator(
        gpt2_model,
        icl_word_list=["dog", "cat", "fish", "bird"],
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
        answer_formatter=first_letter_formatter(),
    )
    # format: dict[letter: (num_true_positives, [split_feature_ids], [probable_feature_absorption_words])]
    likely_negs: dict[str, tuple[int, list[int], list[str]]] = {
        "a": (10, [1, 2, 3], [" Animal", " apple"]),
        "b": (100, [12], [" banana", " bear"]),
    }
    df = calculate_ig_ablation_and_cos_sims(
        calculator, gpt2_l4_sae, fake_probe, likely_negs
    )
    assert df.columns.values.tolist() == snapshot
