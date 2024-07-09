from dataclasses import dataclass


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


def spelling(word: str, separator: str = "-", prefix: str = " ") -> str:
    """
    Break a word string into its component characters, separated by `char_separator`
    e.g. spelling("cat") -> " c-a-t"
    """
    return prefix + separator.join(list(word))


def create_icl_prompt(
    word: str,
    examples: list[str],
    base_template: str = "{word}:",
    char_separator: str = "-",
    spelling_prefix: str = " ",
    example_separator: str = "\n",
) -> SpellingPrompt:
    """
    Create a prompt with ICL examples in the base
    """
    icl_prompts = []
    for ex in examples:
        ex_spelling = spelling(ex, separator=char_separator, prefix=spelling_prefix)
        ex_base = base_template.format(word=ex)
        icl_prompts.append(ex_base + ex_spelling)
    word_spelling = spelling(word, char_separator)
    word_base = base_template.format(word=word)
    return SpellingPrompt(
        base=example_separator.join(icl_prompts) + example_separator + word_base,
        answer=word_spelling,
        word=word,
    )
