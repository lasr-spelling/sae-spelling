from textwrap import dedent

from sae_spelling.prompting import (
    create_icl_prompt,
    first_letter,
    first_letter_formatter,
    spelling,
)


def test_spelling_uses_dash_as_default_separator():
    assert spelling("cat") == " c-a-t"


def test_spelling_can_use_a_custom_separator():
    assert spelling("cat", separator=" :: ") == " c :: a :: t"


def test_spelling_can_capitalize_letter():
    assert spelling("cat", capitalize=True) == " C-A-T"


def test_first_letter_selects_the_first_letter():
    assert first_letter("cat") == " c"


def test_first_letter_can_capitalize_letter():
    assert first_letter("cat", capitalize=True) == " C"


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
