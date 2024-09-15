import torch
from sae_lens import SAE
from syrupy.assertion import SnapshotAssertion

from sae_spelling.experiments.k_sparse_probing import (
    _get_sae_acts,
    calculate_sparse_mean_diff_weights,
    eval_probe_and_sae_k_sparse_raw_scores,
    train_k_sparse_probes,
    train_sparse_multi_probe,
)
from sae_spelling.probing import LinearProbe
from sae_spelling.vocab import LETTERS


def test_train_sparse_multi_probe_results_in_many_zero_weights():
    x = torch.rand(1000, 100)
    y = torch.randint(2, (1000, 3))
    probe1 = train_sparse_multi_probe(x, y, l1_decay=0.01)
    probe2 = train_sparse_multi_probe(x, y, l1_decay=0.1)

    probe1_zero_weights = (probe1.weights.abs() < 1e-5).sum()
    probe2_zero_weights = (probe2.weights.abs() < 1e-5).sum()

    assert probe1_zero_weights > 0
    assert probe2_zero_weights > 0
    assert probe2_zero_weights > probe1_zero_weights


def test_train_k_sparse_probes_returns_reasonable_values(gpt2_l4_sae: SAE):
    train_labels = [("aaa", 0), ("bbb", 1), ("ccc", 2)]
    train_activations = torch.randn(3, 768)
    probes = train_k_sparse_probes(
        gpt2_l4_sae,
        train_labels,
        train_activations,
        ks=[1, 2, 3],
    )
    assert probes.keys() == {1, 2, 3}
    for k, k_probes in probes.items():
        assert k_probes.keys() == {0, 1, 2}
        for probe in k_probes.values():
            assert probe.weight.shape == (k,)
            assert probe.feature_ids.shape == (k,)
            assert probe.k == k


def test_get_sae_acts(gpt2_l4_sae: SAE):
    token_act = torch.randn(768)
    sae_acts = _get_sae_acts(gpt2_l4_sae, token_act.unsqueeze(0), True).squeeze()
    assert sae_acts.shape == (24576,)


def test_get_sae_acts_gives_same_results_batched_and_not_batched(gpt2_l4_sae: SAE):
    token_acts = torch.randn(10, 768)
    sae_acts_unbatched = _get_sae_acts(gpt2_l4_sae, token_acts, True, batch_size=1)
    sae_acts_batched = _get_sae_acts(gpt2_l4_sae, token_acts, True, batch_size=5)
    assert torch.allclose(sae_acts_unbatched, sae_acts_batched)


def test_eval_probe_and_sae_k_sparse_raw_scores_gives_sane_results(
    gpt2_l4_sae: SAE, snapshot: SnapshotAssertion
):
    fake_probe = LinearProbe(768, 26)
    eval_data = [(letter, i) for i, letter in enumerate(LETTERS)]
    eval_activations = torch.randn(26, 768)
    k_sparse_probes = train_k_sparse_probes(
        gpt2_l4_sae,
        eval_data,
        eval_activations,
        ks=[1, 2, 3],
    )
    df = eval_probe_and_sae_k_sparse_raw_scores(
        gpt2_l4_sae,
        fake_probe,
        k_sparse_probes,
        eval_data,
        eval_activations,
    )
    assert df.columns.values.tolist() == snapshot


def test_calculate_sparse_mean_diff_weights_simple():
    x_train = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )  # shape (4, 2)
    y_train = torch.tensor(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ]
    )  # shape (4, 2)

    output = calculate_sparse_mean_diff_weights(
        x_train=x_train,
        y_train=y_train,
        batch_size=2,
        show_progress=False,
    )

    assert torch.allclose(
        output,
        torch.tensor(
            [
                [-2.0, -2.0],
                [2.0, 2.0],
            ]
        ),
    )


def test_calculate_sparse_mean_diff_weights_multiple_classes():
    x_train = torch.tensor(
        [
            [1.0, 20.0],
            [2.0, 30.0],
            [3.0, 40.0],
            [4.0, 50.0],
        ]
    )  # shape (4, 2)
    y_train = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]
    )  # shape (4, 3)

    output = calculate_sparse_mean_diff_weights(
        x_train=x_train,
        y_train=y_train,
        batch_size=2,
        show_progress=False,
    )
    assert torch.allclose(
        output,
        torch.tensor(
            [
                [-1.0, -10.0],
                [1.0, 10.0],
                [2.0, 20.0],
            ]
        ),
    )


def test_calculate_sparse_mean_diff_weights_all_ones():
    # All samples belong to all positive classes
    x_train = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )  # shape (2, 2)

    y_train = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ]
    )  # shape (2, 2)

    output = calculate_sparse_mean_diff_weights(
        x_train=x_train,
        y_train=y_train,
        batch_size=1,
        show_progress=False,
    )
    # It should be assumed that the negative mean is 0 if there are no negative examples
    assert torch.allclose(
        output,
        torch.tensor(
            [
                [2.0, 3.0],
                [2.0, 3.0],
            ]
        ),
    )


def test_calculate_sparse_mean_diff_weights_all_zeros():
    # No samples belong to any positive class
    x_train = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )  # shape (2, 2)

    y_train = torch.tensor(
        [
            [0, 0],
            [0, 0],
        ]
    )  # shape (2, 2)

    output = calculate_sparse_mean_diff_weights(
        x_train=x_train,
        y_train=y_train,
        batch_size=1,
        show_progress=False,
    )

    # It should be assumed that the positive mean is 0 if there are no positive examples
    assert torch.allclose(
        output,
        torch.tensor(
            [
                [-2.0, -3.0],
                [-2.0, -3.0],
            ]
        ),
    )


def test_calculate_sparse_mean_diff_weights_with_unbalanced_classes():
    # Test with a class that has no positive samples
    x_train = torch.tensor(
        [
            [1.0, 20.0],
            [2.0, 30.0],
            [3.0, 40.0],
        ]
    )  # shape (3, 2)
    y_train = torch.tensor(
        [
            [1, 0],
            [1, 1],
            [0, 0],
        ]
    )  # shape (3, 2), second class has no positive samples

    output = calculate_sparse_mean_diff_weights(
        x_train=x_train,
        y_train=y_train,
        batch_size=2,
        show_progress=False,
    )
    assert torch.allclose(
        output,
        torch.tensor(
            [
                [-1.5, -15.0],
                [0.0, 0.0],
            ]
        ),
    )
