import math
import random

import torch
from pytest import approx
from transformer_lens import HookedTransformer

from sae_spelling.prompting import (
    SpellingPrompt,
    create_icl_prompt,
    first_letter_formatter,
    spelling_formatter,
)
from sae_spelling.spelling_grader import Hooks, SpellingGrader


def test_spelling_grader_edits_with_fwd_hooks(gpt2_model: HookedTransformer):
    HOOK_POINT = "blocks.0.hook_resid_post"

    template = "{word} has the first letter:"
    formatter = first_letter_formatter(capitalize=True)

    icl_words = ["stops", "idov", "Atle", "daly"]
    prompt_to_cache = create_icl_prompt(
        "Bon",
        examples=icl_words,
        base_template=template,
        answer_formatter=formatter,
        shuffle_examples=False,
    )

    _, cache = gpt2_model.run_with_cache(prompt_to_cache.base, names_filter=HOOK_POINT)
    cached_act = cache[HOOK_POINT][:, -6, :]  # cache the word position

    def hook_generator(word_indices: list[int]) -> Hooks:
        def hook_fn(act: torch.Tensor, hook: str) -> torch.Tensor:
            _ = hook
            act[:, word_indices, :] = cached_act
            return act

        return [(HOOK_POINT, hook_fn)]

    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=icl_words,
        base_template=template,
        answer_formatter=formatter,
        shuffle_examples=False,
        hook_generator=hook_generator,
    )

    grade_edit = grader.grade_words(["peg", "role"], batch_size=2, use_hooks=True)
    grade_base = grader.grade_words(["peg", "role"], batch_size=2, use_hooks=False)

    assert grade_edit[0].prediction == " B"
    assert grade_edit[0].answer == " P"
    assert grade_base[0].prediction == " P"

    assert grade_edit[1].prediction == " B"
    assert grade_edit[1].answer == " R"
    assert grade_base[1].prediction == " R"


def test_spelling_grader_with_fwd_hooks_intervenes_correct_indices_in_batch(
    gpt2_model: HookedTransformer,
):
    HOOK_POINT = "blocks.0.hook_resid_post"

    word_index_store = {}

    def cache_hook_generator(word_indices: list[int]) -> Hooks:
        word_index_store["cache"] = word_indices.copy()

        def hook_fn(act: torch.Tensor, hook: str) -> torch.Tensor:
            _ = hook
            return act

        return [(HOOK_POINT, hook_fn)]

    template = "{word} is spelled:"
    formatter = spelling_formatter(capitalize=True, separator=" ")
    icl_word_list = ["stops", "bananas", "one", "unobtainable", "two", "three"]
    max_icl_examples = 2

    words = ["peg", "role"]

    random.seed(2)
    prompts = [
        create_icl_prompt(
            word,
            examples=icl_word_list,
            base_template=template,
            answer_formatter=formatter,
            max_icl_examples=max_icl_examples,
            shuffle_examples=True,
        )
        for word in words
    ]

    correct_word_positions = []
    for prompt in prompts:
        prefix_toks = gpt2_model.to_tokens(prompt.base)[0].tolist()
        word_tok = gpt2_model.to_tokens(prompt.word, prepend_bos=False)[0].item()
        correct_word_positions.append(prefix_toks.index(word_tok))

    random.seed(2)
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=icl_word_list,
        base_template=template,
        answer_formatter=formatter,
        max_icl_examples=max_icl_examples,
        shuffle_examples=True,
        hook_generator=cache_hook_generator,
    )

    grader.grade_words(words, batch_size=2, use_hooks=True)

    assert word_index_store["cache"] == correct_word_positions


def test_spelling_grader_marks_response_correct_if_model_predicts_correctly(
    gpt2_model: HookedTransformer,
):
    # GPT2 should be able to spell the word correctly after seeing 4 ICL examples of it spelled correctly
    icl_words = ["correct", "correct", "correct", "correct"]
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=icl_words,
        answer_formatter=spelling_formatter(capitalize=False),
    )
    grade = grader.grade_word("correct")
    assert grade.is_correct
    assert grade.answer == " c-o-r-r-e-c-t"
    assert grade.prediction == " c-o-r-r-e-c-t"
    assert grade.answer_log_prob == grade.prediction_log_prob
    assert (
        grade.prompt
        == create_icl_prompt(
            "correct",
            examples=icl_words,
            answer_formatter=spelling_formatter(capitalize=False),
        ).base
    )


def test_spelling_grader_allows_setting_an_answer_formatter(
    gpt2_model: HookedTransformer,
):
    # GPT2 should be able to spell the word correctly after seeing 4 ICL examples of it spelled correctly
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=["correct", "correct", "correct", "correct"],
        answer_formatter=first_letter_formatter(capitalize=True),
    )
    grade = grader.grade_word("correct")
    assert grade.is_correct
    assert grade.answer == " C"
    assert grade.prediction == " C"
    assert grade.answer_log_prob == grade.prediction_log_prob


def test_spelling_grader_allows_not_shuffling_examples(
    gpt2_model: HookedTransformer,
):
    # GPT2 should be able to spell the word correctly after seeing 4 ICL examples of it spelled correctly
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=["one", "two", "three", "four", "five"],
        shuffle_examples=False,
    )
    grade = grader.grade_word("correct")
    assert grade.prompt.index("one") < grade.prompt.index("two")
    assert grade.prompt.index("two") < grade.prompt.index("three")
    assert grade.prompt.index("three") < grade.prompt.index("four")
    assert grade.prompt.index("four") < grade.prompt.index("five")


def test_spelling_grader_marks_response_wrong_if_model_predicts_incorrectly(
    gpt2_model: HookedTransformer,
):
    # GPT2 is terrible at spelling and will get the following wrong
    grader = SpellingGrader(gpt2_model, icl_word_list=["bananas"], max_icl_examples=1)
    grade = grader.grade_word("incorrect")
    assert grade.is_correct is False
    assert grade.answer == " I-N-C-O-R-R-E-C-T"
    assert grade.prediction != grade.answer
    assert grade.answer_log_prob < grade.prediction_log_prob


def test_spelling_grader_batch_processing_gives_the_same_results_as_individual_processing(
    gpt2_model: HookedTransformer,
):
    grader = SpellingGrader(gpt2_model, icl_word_list=["bananas"], max_icl_examples=1)
    words = ["hotdog", "bananas", "spaceship"]
    batch_grades = grader.grade_words(words, batch_size=len(words))
    individual_grades = [grader.grade_word(word) for word in words]
    for batch_grade, individual_grade in zip(batch_grades, individual_grades):
        assert batch_grade.is_correct == individual_grade.is_correct
        assert batch_grade.answer == individual_grade.answer
        assert batch_grade.prediction == individual_grade.prediction
        assert batch_grade.answer_log_prob == approx(
            individual_grade.answer_log_prob, abs=1e-4
        )
        assert batch_grade.prediction_log_prob == approx(
            individual_grade.prediction_log_prob, abs=1e-4
        )


def test_spelling_grader__grade_response_uses_only_answer_logits_for_log_probs(
    gpt2_model: HookedTransformer,
):
    grader = SpellingGrader(gpt2_model, icl_word_list=[])

    prompt = SpellingPrompt("hi:", " h-i", "hi")
    prompt_len = len(gpt2_model.to_tokens(prompt.base + prompt.answer)[0])
    ans_toks = gpt2_model.to_tokens(prompt.answer, prepend_bos=False)[0].tolist()
    # the answer is 3 tokens long
    assert len(ans_toks) == 3
    vocab_size: int = gpt2_model.tokenizer.vocab_size  # type: ignore
    fake_logits = torch.zeros([prompt_len, vocab_size])
    # set all answer logits to 0% probability
    fake_logits[-4:-1] = -1 * torch.inf
    for i, ans_tok in enumerate(ans_toks):
        # give each answer token a 75% probability
        fake_logits[-3 + i - 1, ans_tok] = math.log(3)
        fake_logits[-3 + i - 1, ans_tok + 1] = 0.0

    grade = grader._grade_response(prompt, fake_logits)
    assert grade.is_correct
    assert grade.answer_log_prob == approx(math.log(0.75**3))
