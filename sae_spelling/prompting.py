import random
from dataclasses import dataclass
from functools import partial
from typing import Callable


@dataclass
class SpellingPrompt:
    """
    Representation of a prompt used for spelling tasks. The prompt consists of a base, and answer, and a word.
    These fields might look like the following:

    base: "The word 'cat' is spelled:"
    answer: " c-a-t"
    word: "cat"

    The base may also contain ICL examples.
    """

    base: str
    answer: str
    word: str


def spelling(
    word: str,
    separator: str = "-",
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> str:
    """
    Break a word string into its component characters, separated by `char_separator`
    e.g. spelling("cat") -> " c-a-t"
    """
    if ignore_leading_space:
        word = word.strip()
    chars = list(word)
    if ignore_non_alpha_chars:
        chars = [c for c in chars if c.isalpha()]
    if capitalize:
        chars = [c.upper() for c in chars]
    return prefix + separator.join(chars)


def first_letter(
    word: str,
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> str:
    """
    return just the first letter of the word, optionally capitalized
    e.g. first_letter("cat") -> " c"
    """
    if ignore_leading_space:
        word = word.strip()
    chars = list(word)
    if ignore_non_alpha_chars:
        chars = [c for c in chars if c.isalpha()]
    first_char = chars[0]
    if capitalize:
        first_char = first_char.upper()
    return prefix + first_char


def last_letter(
    word: str,
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> str:
    """
    return just the last letter of the word, optionally capitalized
    e.g. last_letter("cat") -> " t"
    """
    if ignore_leading_space:
        word = word.strip()
    chars = list(word)
    if ignore_non_alpha_chars:
        chars = [c for c in chars if c.isalpha()]
    last_char = chars[-1]
    if capitalize:
        last_char = last_char.upper()
    return prefix + last_char


def is_present(
    word: str,
    char_to_check: str,
    prefix: str = " ",
    return_binary: bool = False,
) -> str:
    """
    Returns whether a character is present in the word or not
    e.g. is_present("cat", "t") -> "1"
    OR
    is_present("cat", "t") -> "True"
    """
    result = char_to_check in word

    return prefix + str(int(result)) if return_binary else prefix + str(result)


def letter_from_start(
    word: str,
    index: int,
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> str:
    """
    return the letter of the word at the 'index' position relative to the START, optionally capitalized
    e.g. letter_from_start("mobile", 2) -> " b"
    """
    if ignore_leading_space:
        word = word.strip()

    chars = list(word)
    if ignore_non_alpha_chars:
        chars = [c for c in chars if c.isalpha()]

    char_at_idx = chars[index]

    if capitalize:
        char_at_idx = char_at_idx.upper()
    return prefix + char_at_idx


def letter_from_end(
    word: str,
    index: int,
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> str:
    """
    return the letter of the word at the 'index' position relative to the END, optionally capitalized
    NOTE: This follows the Python notation of negative indexing
    i.e mobile[-1] will give 'e' and not 'l'
    e.g. letter_from_end("mobile", 2) -> " i"
    """
    if ignore_leading_space:
        word = word.strip()

    chars = list(word)
    if ignore_non_alpha_chars:
        chars = [c for c in chars if c.isalpha()]

    char_at_idx = chars[-index]

    if capitalize:
        char_at_idx = char_at_idx.upper()
    return prefix + char_at_idx


# ----- Formatters -------------------------------
Formatter = Callable[[str], str]


def spelling_formatter(
    separator: str = "-",
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> Formatter:
    return partial(
        spelling,
        separator=separator,
        prefix=prefix,
        capitalize=capitalize,
        ignore_leading_space=ignore_leading_space,
        ignore_non_alpha_chars=ignore_non_alpha_chars,
    )


def first_letter_formatter(
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> Formatter:
    return partial(
        first_letter,
        prefix=prefix,
        capitalize=capitalize,
        ignore_leading_space=ignore_leading_space,
        ignore_non_alpha_chars=ignore_non_alpha_chars,
    )


def last_letter_formatter(
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> Formatter:
    return partial(
        last_letter,
        prefix=prefix,
        capitalize=capitalize,
        ignore_leading_space=ignore_leading_space,
        ignore_non_alpha_chars=ignore_non_alpha_chars,
    )


def is_present_formatter(
    char_to_check: str,
    return_binary: bool = False,
    prefix: str = " ",
) -> Formatter:
    return partial(
        is_present,
        prefix=prefix,
        char_to_check=char_to_check,
        return_binary=return_binary,
    )


def letter_from_start_formatter(
    index,
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> Formatter:
    return partial(
        letter_from_start,
        index=index,
        prefix=prefix,
        capitalize=capitalize,
        ignore_leading_space=ignore_leading_space,
        ignore_non_alpha_chars=ignore_non_alpha_chars,
    )


def letter_from_end_formatter(
    index,
    prefix: str = " ",
    capitalize: bool = False,
    ignore_leading_space: bool = True,
    ignore_non_alpha_chars: bool = True,
) -> Formatter:
    return partial(
        letter_from_end,
        index=index,
        prefix=prefix,
        capitalize=capitalize,
        ignore_leading_space=ignore_leading_space,
        ignore_non_alpha_chars=ignore_non_alpha_chars,
    )


# --------------------------------


def create_icl_prompt(
    word: str,
    examples: list[str],
    base_template: str = "{word}:",
    example_separator: str = "\n",
    answer_formatter: Formatter = spelling_formatter(),
    max_icl_examples: int | None = None,
    shuffle_examples: bool = True,
) -> SpellingPrompt:
    """
    Create a prompt with ICL examples in the base

    Args:
        word: the word to be spelled
        examples: a list of examples to use as ICL prompts. These will be shuffled
        base_template: a string template for the base of the prompt, including "{word}" as a placeholder for the word
        example_separator: a string to use to separate the ICL examples. default is newline
        answer_formatter: a function to format the answer. default is `spelling_formatter`, which spits out a string like " c-a-t" for the word "cat"
        max_icl_examples: the maximum number of ICL examples to use. If None, all examples will be used. default is None
        shuffle_examples: whether to shuffle the examples before selecting the first `max_icl_examples`. default is True
    """
    icl_prompts = []
    icl_examples = examples

    if max_icl_examples is not None:
        if shuffle_examples:
            icl_examples = random.sample(icl_examples, max_icl_examples)
        else:
            icl_examples = icl_examples[:max_icl_examples]
    elif shuffle_examples:
        icl_examples = examples.copy()
        random.shuffle(icl_examples)
    for ex in icl_examples:
        ex_answer = answer_formatter(ex)
        ex_base = base_template.format(word=ex)
        icl_prompts.append(ex_base + ex_answer)
    word_answer = answer_formatter(word)
    word_base = base_template.format(word=word)
    return SpellingPrompt(
        base=example_separator.join(icl_prompts) + example_separator + word_base,
        answer=word_answer,
        word=word,
    )


def random_icl_prompt(
    vocab: list[str],
    base_template: str = "{word}:",
    example_separator: str = "\n",
    answer_formatter: Formatter = spelling_formatter(),
    max_icl_examples: int = 10,
) -> SpellingPrompt:
    return create_icl_prompt(
        word=random.choice(vocab),
        examples=vocab,
        base_template=base_template,
        example_separator=example_separator,
        answer_formatter=answer_formatter,
        max_icl_examples=max_icl_examples,
    )
