import math

import torch
from pytest import approx
from transformer_lens import HookedTransformer

from sae_spelling.prompting import (
    SpellingPrompt,
    create_icl_prompt,
    first_letter_formatter,
    spelling_formatter,
)
from sae_spelling.spelling_grader import SpellingGrader


def test_spelling_grader_marks_response_correct_if_model_predicts_correctly(
    gpt2_model: HookedTransformer,
):
    # GPT2 should be able to spell the word correctly after seeing 4 ICL examples of it spelled correctly
    icl_words = ["correct", "correct", "correct", "correct"]
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=icl_words,
        answer_formatter=spelling_formatter(capitalize=False),
        check_contamination=False,
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
            check_contamination=False,
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
        check_contamination=False,
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
        check_contamination=False,
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
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=["bananas"],
        max_icl_examples=1,
        check_contamination=False,
    )
    grade = grader.grade_word("incorrect")
    assert grade.is_correct is False
    assert grade.answer == " I-N-C-O-R-R-E-C-T"
    assert grade.prediction != grade.answer
    assert grade.answer_log_prob < grade.prediction_log_prob


def test_spelling_grader_batch_processing_gives_the_same_results_as_individual_processing(
    gpt2_model: HookedTransformer,
):
    grader = SpellingGrader(
        gpt2_model,
        icl_word_list=["bananas"],
        max_icl_examples=1,
        check_contamination=False,
    )
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
    grader = SpellingGrader(gpt2_model, icl_word_list=[], check_contamination=False)

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
