from textwrap import dedent

from transformers import GPT2TokenizerFast

from sae_spelling.prompting import (
    create_icl_prompt,
    first_letter,
    first_letter_formatter,
    is_present,
    is_present_formatter,
    last_letter,
    last_letter_formatter,
    letter_from_end,
    letter_from_end_formatter,
    letter_from_start,
    letter_from_start_formatter,
    spelling,
)
from sae_spelling.vocab import get_alpha_tokens


def test_spelling_uses_dash_as_default_separator():
    assert spelling("cat") == " c-a-t"


def test_spelling_ignores_non_alphanum_chars_and_leading_space_by_default():
    assert spelling("_cat") == " c-a-t"
    assert spelling(" cat") == " c-a-t"
    assert spelling("▁cat") == " c-a-t"
    assert spelling("1cat") == " c-a-t"


def test_spelling_can_respect_non_alphanum_chars():
    assert spelling(" cat", ignore_non_alpha_chars=False) == " c-a-t"
    assert spelling("▁cat", ignore_non_alpha_chars=False) == " ▁-c-a-t"
    assert spelling("1cat", ignore_non_alpha_chars=False) == " 1-c-a-t"


def test_spelling_can_use_a_custom_separator():
    assert spelling("cat", separator=" :: ") == " c :: a :: t"


def test_spelling_can_capitalize_letter():
    assert spelling("cat", capitalize=True) == " C-A-T"


def test_first_letter_selects_the_first_letter():
    assert first_letter("cat") == " c"


def test_first_letter_can_capitalize_letter():
    assert first_letter("cat", capitalize=True) == " C"


def test_first_letter_ignores_non_alphanum_chars_and_leading_space_by_default():
    assert first_letter("_cat") == " c"
    assert first_letter(" cat") == " c"
    assert first_letter(" CAT") == " C"
    assert first_letter("▁cat") == " c"
    assert first_letter("1cat") == " c"


def test_first_letter_can_respect_non_alphanum_chars():
    assert first_letter(" cat", ignore_non_alpha_chars=False) == " c"
    assert first_letter("▁cat", ignore_non_alpha_chars=False) == " ▁"
    assert first_letter("1cat", ignore_non_alpha_chars=False) == " 1"


#---
def test_last_letter_can_capitalize_letter():
    assert last_letter("cat", capitalize=True) == " T"


def test_last_letter_ignores_non_alphanum_chars_and_leading_space_by_default():
    assert last_letter("_cat") == " t"
    assert last_letter(" cat") == " t"
    assert last_letter(" CAT") == " T"
    assert last_letter("▁cat") == " t"
    assert last_letter("1cat") == " t"


def test_last_letter_can_respect_non_alphanum_chars():
    assert last_letter(" cat", ignore_non_alpha_chars=False) == " t"
    assert last_letter("▁cat", ignore_non_alpha_chars=False) == " t"
    assert last_letter("cat1", ignore_non_alpha_chars=False) == " 1"


def test_is_present_can_give_num_binary():
    assert is_present("cat", "a", return_binary=True) == " 1"
    assert is_present("cat", "a", return_binary=False) == " True"


def test_letter_from_start_can_capitalize_letter():
    assert letter_from_start("cat", index = 1, capitalize=True) == " A"


def test_letter_from_start_ignores_non_alphanum_chars_and_leading_space_by_default():
    assert letter_from_start("_cat", index = 1) == " a"
    assert letter_from_start(" cat", index = 1) == " a"
    assert letter_from_start(" CAT", index = 1) == " A"
    assert letter_from_start("▁cat", index = 1) == " a"
    assert letter_from_start("1cat", index = 1) == " a"


def test_letter_from_start_can_respect_non_alphanum_chars():
    assert letter_from_start(" cat", index = 1, ignore_non_alpha_chars=False) == " a"
    assert letter_from_start("▁cat", index = 1, ignore_non_alpha_chars=False) == " c"
    assert letter_from_start("cat1", index = 3, ignore_non_alpha_chars=False) == " 1"


def test_letter_from_end_can_capitalize_letter():
    assert letter_from_end("cat", index = 1, capitalize=True) == " T"


def test_letter_from_end_ignores_non_alphanum_chars_and_leading_space_by_default():
    assert letter_from_end("_cat", index = 1) == " t"
    assert letter_from_end(" cat", index = 1) == " t"
    assert letter_from_end(" CAT", index = 1) == " T"
    assert letter_from_end("▁cat", index = 1) == " t"
    assert letter_from_end("1cat", index = 1) == " t"


def test_letter_from_end_can_respect_non_alphanum_chars():
    assert letter_from_end(" cat", index = 1, ignore_non_alpha_chars=False) == " t"
    assert letter_from_end("▁cat_", index = 1, ignore_non_alpha_chars=False) == " _"
    assert letter_from_end("cat1", index = 1, ignore_non_alpha_chars=False) == " 1"

def test_create_icl_prompt_with_defaults():
    prompt = create_icl_prompt("cat", examples=["dog", "bird"], shuffle_examples=False)

    expected_base = """
        dog: d-o-g
        bird: b-i-r-d
        cat:
    """
    assert prompt.base == dedent(expected_base).strip()
    assert prompt.word == "cat"
    assert prompt.answer == " c-a-t"


def test_create_icl_prompt_with_custom_answer_formatter():
    prompt = create_icl_prompt(
        "cat",
        examples=["dog", "bird"],
        shuffle_examples=False,
        answer_formatter=first_letter_formatter(capitalize=True),
    )

    expected_base = """
        dog: D
        bird: B
        cat:
    """
    assert prompt.base == dedent(expected_base).strip()
    assert prompt.word == "cat"
    assert prompt.answer == " C"


def test_create_icl_prompt_can_specify_max_icl_examples():
    prompt = create_icl_prompt(
        "cat",
        examples=["dog", "bird", "rat", "face"],
        shuffle_examples=False,
        max_icl_examples=1,
    )

    expected_base = """
        dog: d-o-g
        cat:
    """
    assert prompt.base == dedent(expected_base).strip()
    assert prompt.word == "cat"
    assert prompt.answer == " c-a-t"


def test_create_icl_prompt_peformance_is_fast(gpt2_tokenizer: GPT2TokenizerFast):
    "Just run create_icl_prompt lots of times to make sure it's reasonably fast"
    vocab = get_alpha_tokens(gpt2_tokenizer)
    for _ in range(10):
        prompts = [
            create_icl_prompt(word, examples=vocab, max_icl_examples=10)
            for word in vocab
        ]
        assert len(prompts) == len(vocab)
