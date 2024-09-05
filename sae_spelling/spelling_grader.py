from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, cast

import torch
from transformer_lens import HookedTransformer

from sae_spelling.prompting import (
    Formatter,
    SpellingPrompt,
    create_icl_prompt,
    spelling_formatter,
)
from sae_spelling.util import batchify


@dataclass
class SpellingGrade:
    word: str
    prompt: str
    answer: str
    prediction: str
    is_correct: bool
    answer_log_prob: float
    prediction_log_prob: float


HookFn = Callable[[torch.Tensor, str], torch.Tensor]
Hooks = list[tuple[str | Callable, HookFn]]
HookGenerator = Callable[[list[int]], Hooks]


@dataclass
class SpellingGrader:
    model: HookedTransformer
    icl_word_list: list[str]
    max_icl_examples: int | None = None
    base_template: str = "{word}:"
    answer_formatter: Formatter = spelling_formatter()
    example_separator: str = "\n"
    shuffle_examples: bool = True
    hook_generator: HookGenerator | None = None
    check_contamination: bool = True

    def grade_word(self, word: str) -> SpellingGrade:
        return self.grade_words([word])[0]

    def grade_words(
        self,
        words: list[str],
        batch_size: int = 1,
        show_progress: bool = True,
        use_hooks: bool = False,
    ) -> list[SpellingGrade]:
        prompts = [
            create_icl_prompt(
                word,
                examples=self.icl_word_list,
                base_template=self.base_template,
                answer_formatter=self.answer_formatter,
                example_separator=self.example_separator,
                max_icl_examples=self.max_icl_examples,
                shuffle_examples=self.shuffle_examples,
                check_contamination=self.check_contamination,
            )
            for word in words
        ]
        grades = []
        for batch in batchify(prompts, batch_size, show_progress=show_progress):
            inputs = [prompt.base + prompt.answer for prompt in batch]
            if use_hooks:
                assert self.hook_generator is not None
                res = self._run_with_hooks(inputs, batch, self.hook_generator)
            else:
                res = self.model(inputs)
            for i, prompt in enumerate(batch):
                grades.append(self._grade_response(prompt, res[i]))
        return grades

    def _run_with_hooks(
        self,
        inputs: list[str],
        batch: Sequence[SpellingPrompt],
        hook_generator: HookGenerator,
    ) -> torch.Tensor:
        word_indices = []
        for prompt in batch:
            prefix_toks = self.model.to_tokens(prompt.base)[0]
            last_line_tokens = self.model.to_tokens(
                self.base_template.format(word=prompt.word), prepend_bos=False
            )[0]
            last_line_word_index = last_line_tokens.tolist().index(
                self.model.to_tokens(prompt.word, prepend_bos=False)[0].item()
            )
            word_indices.append(
                len(prefix_toks) - len(last_line_tokens) + last_line_word_index
            )

        hooks = hook_generator(word_indices)
        with self.model.hooks(fwd_hooks=hooks):
            return self.model(inputs)

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
            prompt=prompt.base,
            answer=answer,
            prediction=prediction,
            is_correct=is_correct,
            answer_log_prob=answer_tok_log_probs.sum().item(),
            prediction_log_prob=prediction_tok_log_probs.sum().item(),
        )
