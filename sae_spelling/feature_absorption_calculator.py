import random
from dataclasses import dataclass
from typing import Callable

import torch
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from sae_spelling.feature_attribution import (
    calculate_integrated_gradient_attribution_patching,
)
from sae_spelling.prompting import (
    Formatter,
    SpellingPrompt,
    create_icl_prompt,
    first_letter_formatter,
)
from sae_spelling.util import batchify

EPS = 1e-8


@dataclass
class FeatureScore:
    feature_id: int
    activation: float
    ablation_score: float
    probe_cos_sim: float


@dataclass
class AbsorptionResult:
    word: str
    prompt: str
    main_feature_scores: list[FeatureScore]
    top_ablation_feature_scores: list[FeatureScore]
    is_absorption: bool


@dataclass
class SampledAbsorptionResults:
    sample_portion: float
    main_feature_ids: list[int]
    sample_results: list[AbsorptionResult]


@dataclass
class FeatureAbsorptionCalculator:
    """
    Feature absorption calculator for spelling tasks.

    Absorption is defined by the following criteria:
    - The main features for a concept do not fire
    - There is a clear top feature by negative ablation score
    - The top feature is aligned with a probe trained on that concept
    """

    model: HookedTransformer
    icl_word_list: list[str]
    max_icl_examples: int | None = None
    base_template: str = "{word}:"
    answer_formatter: Formatter = first_letter_formatter()
    example_separator: str = "\n"
    shuffle_examples: bool = True
    # the position to read activations from (depends on the template)
    word_token_pos: int = -2
    filter_prompts_batch_size: int = 20
    ig_interpolation_steps: int = 6
    ig_batch_size: int = 6
    topk_feats: int = 10

    # the cosine similarity between the top ablation feature and the probe must be at least this high to count as absorption
    probe_cos_sim_threshold: float = 0.025
    # the top ablation score must be at least this much larger than the second highest score to count as absorption
    ablation_delta_threshold: float = 1.0

    @torch.inference_mode()
    def _filter_prompts(
        self,
        prompts: list[SpellingPrompt],
        sae: SAE,
        main_feature_ids: list[int],
    ) -> list[SpellingPrompt]:
        """
        Filter out any prompts where the main features are already active.
        NOTE: All prompts must have the same token length
        """
        self._validate_prompts_are_same_length(prompts)
        results: list[SpellingPrompt] = []
        for batch in batchify(prompts, batch_size=self.filter_prompts_batch_size):
            sae_in = self.model.run_with_cache([p.base for p in batch])[1][
                sae.cfg.hook_name
            ]
            sae_acts = sae.encode(sae_in)
            split_feats_active = (
                sae_acts[:, self.word_token_pos, main_feature_ids]
                .sum(dim=-1)
                .float()
                .tolist()
            )
            for prompt, res in zip(batch, split_feats_active):
                if res < EPS:
                    results.append(prompt)
        return results

    def _build_prompts(self, words: list[str]) -> list[SpellingPrompt]:
        return [
            create_icl_prompt(
                word,
                examples=self.icl_word_list,
                base_template=self.base_template,
                answer_formatter=self.answer_formatter,
                example_separator=self.example_separator,
                max_icl_examples=self.max_icl_examples,
                shuffle_examples=self.shuffle_examples,
            )
            for word in words
        ]

    def _ig_ablate(
        self,
        sae: SAE,
        prompt: SpellingPrompt,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attrib = calculate_integrated_gradient_attribution_patching(
            model=self.model,
            input=prompt.base,
            include_saes={sae.cfg.hook_name: sae},
            metric_fn=metric_fn,
            patch_indices=self.word_token_pos,
            batch_size=self.ig_batch_size,
            interpolation_steps=self.ig_interpolation_steps,
        )
        with torch.no_grad():
            sae_acts = (
                attrib.sae_feature_activations[sae.cfg.hook_name][
                    0, self.word_token_pos
                ]
                .float()
                .cpu()
                .detach()
                .clone()
            )
            abl_scores = (
                attrib.sae_feature_attributions[sae.cfg.hook_name][
                    0, self.word_token_pos
                ]
                .float()
                .cpu()
                .detach()
                .clone()
            )
            return abl_scores, sae_acts

    def _is_absorption(
        self,
        main_feature_scores: list[FeatureScore],
        top_ablation_feature_scores: list[FeatureScore],
    ) -> bool:
        # if any of the main features fired, this isn't absorption
        if not all(score.activation < EPS for score in main_feature_scores):
            return False
        # if the top ablation score is positive, this isn't absorption
        # NOTE: should we change this? Should we only look at negative ablation scores to begin with?
        if top_ablation_feature_scores[0].ablation_score > 0:
            return False
        # If the top ablation score isn't significantly larger than the next largest score, this isn't absorption
        # NOTE: is this true? Is it possible to have 2 absorbing features firing at the same time?
        act_delta = abs(top_ablation_feature_scores[0].ablation_score) - abs(
            top_ablation_feature_scores[1].ablation_score
        )
        if act_delta < self.ablation_delta_threshold:
            return False
        # If the top firing feature isn't aligned with the probe, this isn't absorption
        if top_ablation_feature_scores[0].probe_cos_sim < self.probe_cos_sim_threshold:
            return False
        return True

    def calculate_absorption_sampled(
        self,
        sae: SAE,
        words: list[str],
        probe_dir: torch.Tensor,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        main_feature_ids: list[int],
        max_ablation_samples: int | None = None,
        filter_prompts: bool = True,
        show_progress: bool = True,
    ) -> SampledAbsorptionResults:
        """
        This method calculates the absorption for each word in the list of words. If `max_ablation_samples` is provided,
        this method will randomly sample that many words to calculate absorption for as a performance optimization.
        If `filter_prompts` is True, this method will filter out any prompts where the main features are already active, as these cannot be absorption.
        """
        prompts = self._build_prompts(words)
        if filter_prompts:
            prompts = self._filter_prompts(prompts, sae, main_feature_ids)
        sample_portion = 1.0
        if max_ablation_samples is not None and len(prompts) > max_ablation_samples:
            sample_portion = max_ablation_samples / len(prompts)
            random.shuffle(prompts)
            prompts = prompts[:max_ablation_samples]
        results: list[AbsorptionResult] = []
        with torch.inference_mode():
            cos_sims = (
                torch.nn.functional.cosine_similarity(
                    probe_dir.to(sae.device), sae.W_dec, dim=-1
                )
                .float()
                .cpu()
            )
        for prompt in tqdm(prompts, disable=not show_progress):
            ig_scores, sae_acts = self._ig_ablate(sae, prompt, metric_fn=metric_fn)
            with torch.inference_mode():
                top_ig_feats = ig_scores.abs().topk(self.topk_feats).indices.tolist()
                main_feature_scores = _get_feature_scores(
                    main_feature_ids,
                    probe_cos_sims=cos_sims,
                    sae_acts=sae_acts,
                    ig_scores=ig_scores,
                )
                top_ablation_feature_scores = _get_feature_scores(
                    top_ig_feats,
                    probe_cos_sims=cos_sims,
                    sae_acts=sae_acts,
                    ig_scores=ig_scores,
                )
                is_absorption = self._is_absorption(
                    top_ablation_feature_scores=top_ablation_feature_scores,
                    main_feature_scores=main_feature_scores,
                )
                results.append(
                    AbsorptionResult(
                        word=prompt.word,
                        prompt=prompt.base,
                        main_feature_scores=main_feature_scores,
                        top_ablation_feature_scores=top_ablation_feature_scores,
                        is_absorption=is_absorption,
                    )
                )
        return SampledAbsorptionResults(
            sample_portion=sample_portion,
            main_feature_ids=main_feature_ids,
            sample_results=results,
        )

    def _validate_prompts_are_same_length(self, prompts: list[SpellingPrompt]):
        "Validate that all prompts have the same token length"
        token_lens = {len(self.model.to_tokens(p.base)[0]) for p in prompts}
        if len(token_lens) > 1:
            raise ValueError(
                "All prompts must have the same token length! Variable-length prompts are not yet supported."
            )


def _get_feature_scores(
    feature_ids: list[int],
    probe_cos_sims: torch.Tensor,
    sae_acts: torch.Tensor,
    ig_scores: torch.Tensor,
) -> list[FeatureScore]:
    return [
        FeatureScore(
            feature_id=feature_id,
            probe_cos_sim=probe_cos_sims[feature_id].item(),
            ablation_score=ig_scores[feature_id].item(),
            activation=sae_acts[feature_id].item(),
        )
        for feature_id in feature_ids
    ]
