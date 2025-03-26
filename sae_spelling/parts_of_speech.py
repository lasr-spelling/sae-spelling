from dataclasses import dataclass
from typing import cast

import nltk
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class TaggedToken:
    """Represents a word with its part-of-speech tag and token information."""

    word: str
    tag: str
    token_positions: list[int]
    char_span: tuple[int, int]


@dataclass
class TaggedSentence:
    """Represents a sentence with its tagged tokens."""

    sentence: str
    tagged_tokens: list[TaggedToken]


@dataclass
class POSMetadata:
    """Metadata for POS activations."""

    pos_tags: list[str]
    pos_indices: dict[str, list[int]]
    tagged_sentences: list[TaggedSentence]


@dataclass
class POSActivations:
    """Container for activations by part of speech."""

    activations: torch.Tensor  # Shape: [num_samples, hidden_dim]
    pos_tags: list[str]  # List of all unique POS tags
    pos_indices: dict[
        str, list[int]
    ]  # Mapping from POS tag to indices in activations tensor
    tagged_sentences: list[TaggedSentence]  # Original tagged sentences


def download_nltk_data() -> None:
    """Download required NLTK data if not already present."""
    nltk.download("treebank")


def reconstruct_sentence(tagged_words: list[tuple[str, str]]) -> str:
    """
    Reconstruct a readable sentence from tagged words using TreebankWordDetokenizer.

    Args:
        tagged_words: list of (word, tag) tuples

    Returns:
        A properly formatted sentence string
    """
    detokenizer = TreebankWordDetokenizer()
    words = [word for word, _ in tagged_words]
    return detokenizer.detokenize(words)


def find_token_positions(
    sentence: str,
    tagged_words: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[TaggedToken]:
    """
    Find token positions for each tagged word in the sentence.

    Args:
        sentence: The reconstructed sentence
        tagged_words: list of (word, tag) tuples
        tokenizer: The tokenizer to use

    Returns:
        list of TaggedToken objects containing word, tag, and token positions
    """
    # Tokenize the full sentence
    tokenized = tokenizer(sentence, return_offsets_mapping=True)
    # Cast to BatchEncoding to help type checker
    tokenized_batch = cast(BatchEncoding, tokenized)

    # Extract input_ids and convert to list to satisfy type checker
    input_ids = tokenized_batch["input_ids"]
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()  # type: ignore

    # Extract offset_mapping and ensure it's a list
    offset_mapping = tokenized_batch["offset_mapping"]
    if not isinstance(offset_mapping, list):
        offset_mapping = offset_mapping.tolist()  # type: ignore

    result = []
    current_pos = 0

    for word, tag in tagged_words:
        # Find the word in the sentence starting from current_pos
        word_start = sentence.find(word, current_pos)
        if word_start == -1:
            # Skip if word not found (shouldn't happen with properly reconstructed sentence)
            continue

        word_end = word_start + len(word)
        current_pos = word_end

        # Find which tokens overlap with this word
        token_positions = []
        for i, (start, end) in enumerate(offset_mapping):
            # Check if this token overlaps with the word
            if end > word_start and start < word_end:
                token_positions.append(i)

        result.append(
            TaggedToken(
                word=word,
                tag=tag,
                token_positions=token_positions,
                char_span=(word_start, word_end),
            )
        )

    return result


def get_treebank_tagged_sents() -> list[list[tuple[str, str]]]:
    """
    Get tagged sentences from the NLTK treebank corpus.

    Returns:
        List of sentences, where each sentence is a list of (word, tag) tuples
    """
    # Ensure treebank data is downloaded
    download_nltk_data()
    return nltk.corpus.treebank.tagged_sents()


def create_pos_dataset(
    tokenizer: PreTrainedTokenizerBase,
    tagged_sents: list[list[tuple[str, str]]],
) -> list[TaggedSentence]:
    """
    Create a dataset from tagged sentences.

    Args:
        tokenizer: The tokenizer to use
        tagged_sents: Optional list of tagged sentences. If None, uses NLTK treebank.

    Returns:
        list of TaggedSentence objects containing sentence and tagged token information
    """

    dataset = []
    for tagged_words in tagged_sents:
        # Reconstruct the sentence using TreebankWordDetokenizer
        sentence = reconstruct_sentence(tagged_words)

        # Find token positions for each word
        tagged_tokens = find_token_positions(sentence, tagged_words, tokenizer)

        dataset.append(TaggedSentence(sentence=sentence, tagged_tokens=tagged_tokens))

    return dataset
