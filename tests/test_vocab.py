from transformers import GPT2TokenizerFast

from sae_spelling.vocab import (
    LETTERS,
    LETTERS_UPPER,
    get_alpha_tokens,
    get_brown_words,
    get_common_word_tokens,
    get_common_words,
    get_nltk_words,
    get_same_ending_word_pairs_of_len,
    get_tokens,
)


def is_alpha(word: str) -> bool:
    return all(char in LETTERS or char in LETTERS_UPPER for char in word)


def test_get_tokens_returns_all_tokens_by_default(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    tokens = get_tokens(gpt2_tokenizer)
    assert len(tokens) == len(gpt2_tokenizer.vocab)


def test_get_tokens_can_keep_special_chars(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    tokens = get_tokens(gpt2_tokenizer, replace_special_chars=False)
    assert not any(token.startswith(" ") for token in tokens)
    assert any(token.startswith("_") for token in tokens)


def test_get_tokens_replaces_special_chars_by_default(
    gpt2_tokenizer: GPT2TokenizerFast,
):
    tokens = get_tokens(gpt2_tokenizer)
    assert any(token.startswith(" ") for token in tokens)


def test_get_tokens_filter_returned_tokens(gpt2_tokenizer: GPT2TokenizerFast):
    tokens = get_tokens(
        gpt2_tokenizer,
        lambda token: token.isalpha() and token.isupper(),
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


def test_get_brown_words():
    words = get_brown_words()
    assert len(words) == 1161192


def test_get_common_words():
    nltk_words = set(word for word in get_nltk_words())
    words = get_common_words()
    assert all(word in nltk_words for word in words.keys())

    threshold = 50
    more_restricted_words = get_common_words(threshold=threshold)
    assert all(
        more_restricted_words[word] >= threshold for word in more_restricted_words
    )
    assert all(word in nltk_words for word in more_restricted_words.keys())
    assert len(more_restricted_words.keys()) < len(words.keys())


def test_get_same_ending_word_pairs_of_len():
    length = 5
    pairs = get_same_ending_word_pairs_of_len(length)

    assert all(len(word1) == length for word1, word2 in pairs)
    assert all(word1.islower() and word2.islower() for word1, word2 in pairs)
    assert all(word1[1:] == word2[1:] for word1, word2 in pairs)


def test_get_common_word_tokens(gpt2_tokenizer: GPT2TokenizerFast):
    threshold = 10
    common_words = set(get_common_words(threshold=threshold).keys())
    tokens = get_common_word_tokens(
        gpt2_tokenizer,
        threshold=threshold,
        only_start_of_word=True,
        replace_special_chars=True,
    )
    assert all(token.startswith(" ") for token in tokens)
    assert all(token[1:] in common_words for token in tokens)
