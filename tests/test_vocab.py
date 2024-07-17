from transformers import GPT2TokenizerFast

from sae_spelling.vocab import (
    LETTERS_UPPER,
    get_alpha_tokens,
    get_nltk_words,
    get_tokens,
)


def is_alpha(word: str) -> bool:
    return all(char.upper() in LETTERS_UPPER for char in word)


def test_get_tokens_returns_all_tokens_by_default(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    tokens = get_tokens(gpt2_tokenizer)
    assert len(tokens) == len(gpt2_tokenizer.vocab)


def test_get_tokens_keeps_special_chars_by_default(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    tokens = get_tokens(gpt2_tokenizer)
    assert not any(token.startswith(" ") for token in tokens)
    assert any(token.startswith("_") for token in tokens)


def test_get_tokens_can_filter_returned_tokens(gpt2_tokenizer: GPT2TokenizerFast):
    tokens = get_tokens(
        gpt2_tokenizer,
        lambda token: token.isalpha() and token.isupper(),
        replace_special_chars=True,
    )
    assert all(token.isalpha() and token.isupper() for token in tokens)


def test_get_alpha_tokens_includes_leading_spaces_by_default(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    alpha_tokens = get_alpha_tokens(gpt2_tokenizer, replace_special_chars=True)
    assert any(token.startswith(" ") for token in alpha_tokens)
    assert all(is_alpha(token.strip()) for token in alpha_tokens)
    assert all(token.strip().isalpha() for token in alpha_tokens)


def test_get_alpha_tokens_can_remove_leading_spaces(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    alpha_tokens = get_alpha_tokens(
        gpt2_tokenizer, allow_leading_space=False, replace_special_chars=True
    )
    assert all(token.isalpha() for token in alpha_tokens)


def test_get_nltk_words():
    words = get_nltk_words()
    assert len(words) == 236736
