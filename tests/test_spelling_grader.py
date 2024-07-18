import math

import torch
from pytest import approx
from transformer_lens import HookedTransformer

from sae_spelling.prompting import SpellingPrompt
from sae_spelling.spelling_grader import SpellingGrader


def test_spelling_grader_marks_response_correct_if_model_predicts_correctly(
    gpt2_model: HookedTransformer,
):
    # GPT2 should be able to spell the word correctly after seeing 4 ICL examples of it spelled correctly
    grader = SpellingGrader(
        gpt2_model, icl_word_list=["correct", "correct", "correct", "correct"]
    )
    grade = grader.grade_word("correct")
    assert grade.is_correct
    assert grade.answer == " c-o-r-r-e-c-t"
    assert grade.prediction == " c-o-r-r-e-c-t"
    assert grade.answer_log_prob == grade.prediction_log_prob


def test_spelling_grader_marks_response_wrong_if_model_predicts_incorrectly(
    gpt2_model: HookedTransformer,
):
    # GPT2 is terrible at spelling and will get the following wrong
    grader = SpellingGrader(gpt2_model, icl_word_list=["bananas"], max_icl_examples=1)
    grade = grader.grade_word("incorrect")
    assert grade.is_correct is False
    assert grade.answer == " i-n-c-o-r-r-e-c-t"
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
        assert batch_grade.answer_log_prob == approx(individual_grade.answer_log_prob, abs=1e-5)
        assert batch_grade.prediction_log_prob == approx(
            individual_grade.prediction_log_prob,
            abs=1e-5
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
