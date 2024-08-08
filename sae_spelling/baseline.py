import random
import string
from collections.abc import Generator
from typing import Any

import torch
from tqdm.autonotebook import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from sae_spelling.prompting import create_icl_prompt, spelling_formatter


def get_valid_vocab(
    tokenizer: PreTrainedTokenizerFast, sample_cutoff: int = 1000
) -> dict:
    """
    This function takes in a model tokenizer, and then returns a dictionary of valid tokens to spell from the tokenizer organised by length of token (leading spaces/underscores).

    Args:
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    sample_cutoff (int): Minimum number of tokens that must have the given length for that length to be chosen for testing.

    Outputs:
    dict: A dictionary containing each a token length, and then the collection of tokens that are this long.
    """
    full_vocab = []
    valid_chars = set(string.ascii_letters)
    for token in tokenizer.vocab.keys():
        word = tokenizer.convert_tokens_to_string([token])
        if (word[0] == " ") | (word[0] == "▁"):
            word = word[1:]
        if len(word) > 1 and all(char in valid_chars for char in word):
            full_vocab.append(word)
    full_vocab = list(
        set(full_vocab)
    )  # removes duplicate after removing leading spaces
    total_vocab_dict = {}
    long_token_cutoff = len(
        max(full_vocab, key=len)
    )  # find the length of longest token

    for i in range(1, long_token_cutoff + 1):
        _tmplist = [s for s in full_vocab if len(s) == i]
        if len(_tmplist) == 0:
            continue
        total_vocab_dict[str(i)] = _tmplist

    keys_to_delete = [
        i for i in total_vocab_dict.keys() if len(total_vocab_dict[i]) < sample_cutoff
    ]

    for key in keys_to_delete:
        del total_vocab_dict[key]

    return total_vocab_dict


def generate_and_score_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    vocab_dict: dict,
    samples_per_combo: int,
    max_icl_length: int = 8,
    capitals: str | None = None,
    char_gap: str = "-",
    example_gap: str = " ",
    batch_size: int = 32,
) -> Generator[dict[str, Any | int | float], Any, Any]:
    """
    This function takes in various user parameters, iterating over different word lengths and in-context learning (ICL) lengths,
    calculates accuracy scores for each batch and then outputs them to a dict. This can then be turned into a dataframe for plotting and analysis.

    Args:
    model (transformers.PreTrainedModel): The model to use for generation (assumed to be downloaded from Huggingface)
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    vocab_dict (dict): The dictionary of tokens to test for (assumed to be created by the get_valid_vocab function above)
    samples_per_combo (int): Number of samples to generate for each word length x ICL length combination.
    max_icl_length (int, optional): Maximum in-context learning length. Defaults to 8.
    capitals (str, optional): Capitalization style ('upper', 'lower', or None for original case). Defaults to None.
    char_gap (str, optional): Character to use as separator in spellings. Defaults to '-'.
    example_gap (str, optional): Separator between examples in ICL prompts. Defaults to ' '.
    batch_size (int, optional): Batch size for processing. Defaults to 32.

    Yields:
    dict: A dictionary containing 'word_length', 'icl_length', and 'accuracy' for each combination.
    """
    total_combinations = len(vocab_dict) * max_icl_length
    with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
        for word_length in vocab_dict.keys():
            words = vocab_dict[word_length]
            tokens_to_gen = (
                int(word_length) * 2 - 1 if char_gap != " " else int(word_length)
            )

            for icl_length in range(1, max_icl_length + 1):
                test_vocab = random.sample(
                    words, k=min(samples_per_combo, len(words) - max_icl_length)
                )
                sample_vocab = [w for w in words if w not in test_vocab]
                all_correct = 0
                total_processed = 0

                for i in range(0, len(test_vocab), batch_size):
                    batch = test_vocab[i : i + batch_size]
                    inputs = []
                    targets = []
                    for w in batch:
                        test_case = create_icl_prompt(
                            [
                                w.upper()
                                if capitals == "upper"
                                else w.lower()
                                if capitals == "lower"
                                else w
                            ][0],
                            examples=sample_vocab,
                            example_separator=example_gap,
                            answer_formatter=spelling_formatter(
                                separator=char_gap,
                                capitalize=[True if capitals == "upper" else False][0],
                            ),
                            max_icl_examples=icl_length,
                        )
                        inputs.append(
                            [
                                test_case.base.upper()
                                if capitals == "upper"
                                else test_case.base.lower()
                                if capitals == "lower"
                                else test_case.base
                            ][0]
                        )
                        targets.append(test_case.answer)

                    # Process batch
                    with torch.no_grad():
                        input_ids = tokenizer(
                            inputs, padding=True, truncation=True, return_tensors="pt"
                        ).to("cuda")
                        input_length = input_ids["input_ids"].shape[1]  # type: ignore
                        outputs = model.generate(
                            **input_ids,  # type: ignore
                            max_new_tokens=tokens_to_gen,  # type: ignore
                        )
                        answers = tokenizer.batch_decode(
                            outputs[:, input_length:], skip_special_tokens=True
                        )
                    all_correct += sum(a == t for a, t in zip(answers, targets))
                    total_processed += len(batch)
                    torch.cuda.empty_cache()

                accuracy = all_correct / total_processed

                yield {
                    "word_length": word_length,
                    "icl_length": icl_length,
                    "accuracy": accuracy,
                }

                pbar.update(1)
