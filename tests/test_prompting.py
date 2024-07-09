from textwrap import dedent

from sae_spelling.prompting import create_icl_prompt, spelling


def test_spelling_uses_dash_as_default_separator():
    assert spelling("cat") == " c-a-t"


def test_spelling_can_use_a_custom_separator():
    assert spelling("cat", separator=" :: ") == " c :: a :: t"


def test_create_icl_prompt_with_default_separators():
    prompt = create_icl_prompt("cat", examples=["dog", "bird"])

    expected_base = """
        dog: d-o-g
        bird: b-i-r-d
        cat:
    """
    assert prompt.base == dedent(expected_base).strip()
    assert prompt.word == "cat"
    assert prompt.answer == " c-a-t"
