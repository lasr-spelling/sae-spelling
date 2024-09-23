import torch
from sae_lens import SAE
from syrupy.assertion import SnapshotAssertion

from sae_spelling.experiments.latent_evaluation import (
    eval_probe_and_top_sae_raw_scores,
)
from sae_spelling.probing import LinearProbe


@torch.no_grad()
def test_eval_probe_and_top_sae_raw_scores_gives_sane_results(
    gpt2_l4_sae: SAE, snapshot: SnapshotAssertion
):
    fake_probe = LinearProbe(768, 26)
    fake_probe.weights[2, :] = gpt2_l4_sae.W_enc[:, 123]  # set C to feature 123

    eval_data = [("dog", 3), ("cat", 2), ("fish", 5), ("bird", 1)]
    eval_activations = torch.randn(4, 768)
    df = eval_probe_and_top_sae_raw_scores(
        gpt2_l4_sae,
        fake_probe,
        eval_data,
        eval_activations,
        metadata={"layer": 4},
        topk=2,
    )
    assert all(df["sae_c_top_0_feat"] == 123)
    assert df.columns.values.tolist() == snapshot
