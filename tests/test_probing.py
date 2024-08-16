import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from torch.nn.functional import cosine_similarity, one_hot

from sae_spelling.probing import (
    LinearProbe,
    ProbeStats,
    SpellingPrompt,
    _calc_pos_weights,
    _get_exponential_decay_scheduler,
    create_dataset_probe_training,
    gen_and_save_df_acts_probing,
    gen_probe_stats,
    load_df_acts_probing,
    train_binary_probe,
    train_linear_probe_for_task,
    train_multi_probe,
)


def test_get_exponential_decay_scheduler_decays_from_lr_to_end_lr_over_num_epochs():
    optim = torch.optim.Adam([torch.zeros(1)], lr=0.01)
    scheduler = _get_exponential_decay_scheduler(
        optim, start_lr=0.01, end_lr=1e-5, num_steps=100
    )
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    assert lrs[0] == pytest.approx(0.01, abs=1e-6)
    assert lrs[-1] == pytest.approx(1e-5, abs=1e-6)


def test_calc_pos_weights_returns_1_for_equal_weights():
    y_train = torch.cat([torch.ones(5), torch.zeros(5)]).unsqueeze(1)
    pos_weights = _calc_pos_weights(y_train)
    assert torch.allclose(pos_weights, torch.tensor([1.0]))


def test_calc_pos_weights_is_above_1_if_the_are_more_neg_than_pos_samples():
    y_train = torch.cat([torch.ones(5), torch.zeros(15)]).unsqueeze(1)
    pos_weights = _calc_pos_weights(y_train)
    assert torch.allclose(pos_weights, torch.tensor([3.0]))


def test_calc_pos_weights_is_below_1_if_the_are_more_neg_than_pos_samples():
    y_train = torch.cat([torch.ones(20), torch.zeros(5)]).unsqueeze(1)
    pos_weights = _calc_pos_weights(y_train)
    assert torch.allclose(pos_weights, torch.tensor([0.25]))


def test_calc_pos_weights_handles_multiclass_samples():
    y_train = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 0],
        ]
    )
    pos_weights = _calc_pos_weights(y_train)
    assert torch.allclose(pos_weights, torch.tensor([1.0, 0.2]))


@pytest.mark.parametrize("seed", range(5))
def test_train_binary_probe_scores_highly_on_fully_separable_datasets(seed):
    torch.manual_seed(seed)
    neg_center = 5.0 * torch.ones(64)
    pos_center = -5.0 * torch.ones(64)

    neg_xs = neg_center + torch.randn(1024, 64)
    pos_xs = pos_center + torch.randn(368, 64)

    x = torch.cat([neg_xs, pos_xs])
    y = torch.cat([torch.zeros(len(neg_xs)), torch.ones(len(pos_xs))])

    # shuffle x and y in the same way
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    x_train = x[:768]
    y_train = y[:768]

    x_test = x[768:]
    y_test = y[768:]

    probe = train_binary_probe(
        x_train,
        y_train,
        num_epochs=100,
        batch_size=128,
        weight_decay=1e-7,
        lr=1.0,
        show_progress=False,
    )

    train_preds = probe(x_train)[:, 0] > 0
    test_preds = probe(x_test)[:, 0] > 0

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    assert train_acc > 0.98
    assert test_acc > 0.98

    sk_probe = LogisticRegression(max_iter=100, class_weight="balanced").fit(
        x_train.numpy(), y_train.numpy()
    )

    # since this is a synthetic dataset, we know the correct direction we should learn
    correct_dir = (pos_center - neg_center).unsqueeze(0)
    # just verify that sklearn does get the right answer
    sk_cos_sim = cosine_similarity(correct_dir, torch.tensor(sk_probe.coef_), dim=1)
    assert sk_cos_sim.min().item() > 0.98
    cos_sim = cosine_similarity(correct_dir, probe.weights, dim=1)
    assert cos_sim.min().item() > 0.98


@pytest.mark.parametrize("seed", range(5))
def test_train_binary_probe_scores_highly_on_noisy_datasets(seed):
    torch.manual_seed(seed)
    neg_center = 1.0 * torch.ones(64)
    pos_center = -1.0 * torch.ones(64)

    neg_xs = neg_center + 3 * torch.randn(1024, 64)
    pos_xs = pos_center + 3 * torch.randn(368, 64)

    x = torch.cat([neg_xs, pos_xs])
    y = torch.cat([torch.zeros(len(neg_xs)), torch.ones(len(pos_xs))])

    # shuffle x and y in the same way
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    x_train = x[:600]
    y_train = y[:600]

    x_test = x[600:]
    y_test = y[600:]

    probe = train_binary_probe(
        x_train,
        y_train,
        num_epochs=100,
        batch_size=128,
        weight_decay=1e-7,
        lr=1.0,
        show_progress=False,
    )

    train_preds = probe(x_train)[:, 0] > 0
    test_preds = probe(x_test)[:, 0] > 0

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    assert train_acc > 0.9
    assert test_acc > 0.9

    sk_probe = LogisticRegression(max_iter=100, class_weight="balanced").fit(
        x_train.numpy(), y_train.numpy()
    )

    # since this is a synthetic dataset, we know the correct direction we should learn
    correct_dir = (pos_center - neg_center).unsqueeze(0)
    # just verify that sklearn does get the right answer
    sk_cos_sim = cosine_similarity(correct_dir, torch.tensor(sk_probe.coef_), dim=1)
    print(f"sklearn cos sim: {sk_cos_sim}")
    assert sk_cos_sim.min().item() > 0.85
    cos_sim = cosine_similarity(correct_dir, probe.weights, dim=1)
    assert cos_sim.min().item() > 0.85


@pytest.mark.parametrize("seed", range(5))
def test_train_multi_probe_scores_highly_on_fully_separable_datasets(seed):
    torch.manual_seed(seed)
    class1_center = 10 * torch.randn(64)
    class2_center = 10 * torch.randn(64)
    class3_center = 10 * torch.randn(64)

    class1_xs = class1_center + torch.randn(128, 64)
    class2_xs = class2_center + torch.randn(32, 64)
    class3_xs = class3_center + torch.randn(200, 64)

    x = torch.cat([class1_xs, class2_xs, class3_xs])
    y = one_hot(
        torch.cat(
            [
                0 * torch.ones(len(class1_xs)),  # 0
                1 * torch.ones(len(class2_xs)),  # 1
                2 * torch.ones(len(class3_xs)),  # 2
            ]
        ).long()
    )

    # shuffle x and y in the same way
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    x_train = x[:300]
    y_train = y[:300]

    x_test = x[300:]
    y_test = y[300:]

    probe = train_multi_probe(
        x_train, y_train, num_probes=3, num_epochs=100, batch_size=32
    )

    train_preds = probe(x_train) > 0
    test_preds = probe(x_test) > 0

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    assert train_acc > 0.98
    assert test_acc > 0.98


@pytest.mark.parametrize("seed", range(5))
def test_train_multi_probe_scores_highly_on_noisy_datasets(seed):
    torch.manual_seed(seed)
    class1_center = 0.5 * torch.randn(64)
    class2_center = 0.5 * torch.randn(64)
    class3_center = 0.5 * torch.randn(64)

    class1_xs = class1_center + torch.randn(150, 64)
    class2_xs = class2_center + torch.randn(600, 64)
    class3_xs = class3_center + torch.randn(250, 64)

    x = torch.cat([class1_xs, class2_xs, class3_xs])
    y = one_hot(
        torch.cat(
            [
                0 * torch.ones(len(class1_xs)),  # 0
                1 * torch.ones(len(class2_xs)),  # 1
                2 * torch.ones(len(class3_xs)),  # 2
            ]
        ).long()
    )

    # shuffle x and y in the same way
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    x_train = x[:500]
    y_train = y[:500]

    x_test = x[500:]
    y_test = y[500:]

    probe = train_multi_probe(
        x_train, y_train, num_probes=3, num_epochs=100, batch_size=128
    )

    train_preds = probe(x_train) > 0
    test_preds = probe(x_test) > 0

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    assert train_acc > 0.9
    assert test_acc > 0.9


def test_create_dataset_probe_training():
    vocab = ["apple", "banana", "cherry"]
    formatter = lambda x: x[0].upper()
    num_prompts_per_token = 2
    max_icl_examples = 2

    result = create_dataset_probe_training(
        vocab, formatter, num_prompts_per_token, max_icl_examples
    )

    assert len(result) == len(vocab) * num_prompts_per_token
    for prompt, answer_class in result:
        assert isinstance(prompt, SpellingPrompt)
        assert isinstance(answer_class, int)
        assert prompt.answer in ["A", "B", "C"]
        assert 0 <= answer_class <= 25


@patch("sae_spelling.probing.HookedTransformer")
@patch("sae_spelling.probing.pd.DataFrame.to_csv")
def test_gen_and_save_df_acts_probing(mock_to_csv, mock_model):
    dataset = [
        (
            SpellingPrompt(
                base="The word 'cat' is spelled:", answer=" c-a-t", word="cat"
            ),
            0,
        ),
        (
            SpellingPrompt(
                base="The word 'dog' is spelled:", answer=" d-o-g", word="dog"
            ),
            1,
        ),
    ]
    mock_model.cfg.d_model = 768
    mock_cache = {
        "test_hook": torch.rand(2, 10, 768)  # 2 samples, 10 tokens, 768 dimensions
    }
    mock_model.run_with_cache.return_value = (None, mock_cache)

    # temporary directory for memmap
    with tempfile.TemporaryDirectory() as tmpdir:
        df, memmap = gen_and_save_df_acts_probing(
            mock_model,
            dataset,
            tmpdir,
            "test_hook",
            "test_task",
            layer=0,
            batch_size=2,
            position_idx=-2,
        )

        assert isinstance(df, pd.DataFrame)
        assert isinstance(memmap, np.memmap)
        assert mock_to_csv.called

        # Check if the memmap file was created
        memmap_path = os.path.join(tmpdir, "test_task", "test_task_act_tensor.dat")
        assert os.path.exists(memmap_path)
        assert memmap.shape == (2, 768)  # 2 samples, 768 dimensions

    # the temporary directory is autocleaned here


@patch("sae_spelling.probing.pd.read_csv")
def test_load_df_acts_probing(mock_read_csv):
    mock_model = Mock()
    mock_model.cfg.d_model = 768

    mock_df = pd.DataFrame({"index": range(100)})
    mock_read_csv.return_value = mock_df

    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = os.path.join(tmpdir, "test_task")
        os.makedirs(task_dir)

        memmap_path = os.path.join(task_dir, "test_task_act_tensor.dat")
        memmap = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(100, 768))
        memmap[:] = np.random.rand(100, 768).astype(np.float32)
        memmap.flush()

        df, acts = load_df_acts_probing(mock_model, tmpdir, "test_task", layer=0)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(acts, np.memmap)
        assert df.shape[0] == acts.shape[0]
        assert acts.shape == (100, 768)


def test_train_linear_probe_for_task():
    task_df = pd.DataFrame({"answer_class": np.random.randint(0, 26, 1000)})
    device = torch.device("cpu")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_filename = tmp.name

    task_act_tensor = np.memmap(
        tmp_filename, dtype="float32", mode="w+", shape=(1000, 768)
    )
    task_act_tensor[:] = np.random.rand(1000, 768).astype(np.float32)
    task_act_tensor.flush()

    probe, probe_data = train_linear_probe_for_task(
        task_df, device, task_act_tensor, num_classes=26, batch_size=32, num_epochs=2
    )

    assert isinstance(probe, LinearProbe)
    assert isinstance(probe_data, dict)
    assert all(
        key in probe_data
        for key in ["X_train", "X_val", "y_train", "y_val", "train_idx", "val_idx"]
    )

    # Clean up
    del task_act_tensor
    os.unlink(tmp_filename)


def test_gen_probe_stats():
    probe = LinearProbe(input_dim=768, num_outputs=26)
    X_val = torch.rand(100, 768)
    y_val = torch.randint(0, 26, (100,))

    results = gen_probe_stats(probe, X_val, y_val)

    assert len(results) == 26
    for result in results:
        assert isinstance(result, ProbeStats)
        assert isinstance(result.letter, str)
        assert all(
            isinstance(getattr(result, key), float)
            for key in ["f1", "accuracy", "precision", "recall"]
        )
