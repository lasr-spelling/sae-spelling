import random
from dataclasses import dataclass
from typing import cast

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_spelling.prompting import SpellingPrompt, create_icl_prompt
from sae_spelling.util import batchify


@dataclass
class SpellingGrade:
    word: str
    answer: str
    prediction: str
    is_correct: bool
    answer_log_prob: float
    prediction_log_prob: float


@dataclass
class SpellingGrader:
    model: HookedTransformer
    icl_word_list: list[str]
    n_icl_examples: int = 4
    base_template: str = "{word}:"
    char_separator: str = "-"
    spelling_prefix: str = " "
    example_separator: str = "\n"

    def grade_word(self, word: str) -> SpellingGrade:
        return self.grade_words([word])[0]

    def grade_words(self, words: list[str], batch_size: int = 1) -> list[SpellingGrade]:
        prompts = [
            create_icl_prompt(
                word,
                random.sample(self.icl_word_list, self.n_icl_examples),
                base_template=self.base_template,
                char_separator=self.char_separator,
                spelling_prefix=self.spelling_prefix,
                example_separator=self.example_separator,
            )
            for word in words
        ]
        grades = []
        for batch in tqdm(batchify(prompts, batch_size)):
            inputs = [prompt.base + prompt.answer for prompt in batch]
            res = self.model(inputs)
            for i, prompt in enumerate(batch):
                grades.append(self._grade_response(prompt, res[i]))
        return grades

    def _grade_response(
        self, prompt: SpellingPrompt, model_output: torch.Tensor
    ) -> SpellingGrade:
        tokenizer = self.model.tokenizer
        assert tokenizer is not None
        prefix_toks = self.model.to_tokens(prompt.base)[0]
        answer_toks = self.model.to_tokens(prompt.answer, prepend_bos=False)[0]
        relevant_logits = model_output[
            len(prefix_toks) - 1 : len(prefix_toks) + len(answer_toks) - 1
        ]
        log_probs = torch.nn.functional.log_softmax(relevant_logits, dim=-1)
        prediction_toks = torch.argmax(log_probs, dim=-1)
        answer_tok_log_probs = log_probs.gather(1, answer_toks.unsqueeze(-1))
        prediction_tok_log_probs = log_probs.gather(1, prediction_toks.unsqueeze(-1))
        is_correct = cast(bool, torch.all(prediction_toks == answer_toks).item())
        answer = tokenizer.decode(answer_toks)
        prediction = tokenizer.decode(prediction_toks)
        return SpellingGrade(
            word=prompt.word,
            answer=answer,
            prediction=prediction,
            is_correct=is_correct,
            answer_log_prob=answer_tok_log_probs.sum().item(),
            prediction_log_prob=prediction_tok_log_probs.sum().item(),
        )
