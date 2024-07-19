from transformer_lens import HookedTransformer
import torch
from typing import List, Tuple

def intervention_hook(
    sub_vecs: List[torch.Tensor],
    add_vecs: List[torch.Tensor],
    intervention_position: int,
    force_add_magnitude: float | None = None
):
    intervention_applied = False

    def hook(act: torch.Tensor, hook):
        nonlocal intervention_applied

        if not intervention_applied:
            for sub_vec, add_vec in zip(sub_vecs, add_vecs):
                vec = sub_vec.to(torch.bfloat16)
                to_vec = add_vec.to(torch.bfloat16)

                vec_normed = vec / vec.norm(dim=-1, keepdim=True)
                to_vec_normed = to_vec / to_vec.norm(dim=-1, keepdim=True)

                proj = act[:, intervention_position] @ vec_normed

                if force_add_magnitude is None:
                    act[:, intervention_position, :] -= proj.unsqueeze(-1) * vec_normed
                    act[:, intervention_position, :] += proj.unsqueeze(-1) * to_vec_normed
                else:
                    act[:, intervention_position, :] -= proj.unsqueeze(-1) * vec_normed
                    act[:, intervention_position, :] += force_add_magnitude * to_vec_normed

            intervention_applied = True

        return act

    return hook

def run_model_with_edits(
        model: HookedTransformer, 
        prompt: str, 
        subtracted_vectors: List[torch.Tensor], 
        added_vectors: List[torch.Tensor], 
        hook_point: str, 
        num_new_tokens: int = 15,
        force_add_magnitude: float | None = None
    ) -> Tuple[str, str]:
    """
    Runs the model with a number of edits, subtracting and adding vectors to the embedding of the token at `edit_token_position`.
    The model is run with and without the hook, and the generated tokens are returned.
    """

    prompt_tokens = model.to_tokens(prompt, prepend_bos = True)
    token_position = prompt_tokens.shape[1] - 2 # the token before a colon in a typical ICL prompt

    prediction_without_edits = model.generate(prompt, max_new_tokens=num_new_tokens, verbose=False)[len(prompt):]

    with model.hooks(fwd_hooks=[(hook_point, intervention_hook(subtracted_vectors, added_vectors, token_position, force_add_magnitude))]):
        prediction_with_edits = model.generate(prompt, max_new_tokens=num_new_tokens, verbose=False)[len(prompt):]

    return prediction_without_edits, prediction_with_edits