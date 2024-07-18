from collections import Counter

import torch
from sklearn.linear_model import LogisticRegression

from sae_spelling.probing import create_rebalanced_dataloader, train_multiclass_probe


def test_create_rebalanced_dataloader_balances_class_frequencies():
    x = torch.randn(10_000, 10)
    weights = torch.tensor([1, 1, 5, 10, 1])
    probs = weights / weights.sum()
    y = torch.multinomial(probs, 10_000, replacement=True)
    loader = create_rebalanced_dataloader(x, y, len(weights), batch_size=10)
    counts = Counter()
    for _, batch_y in loader:
        assert len(batch_y) == 10
        counts.update(batch_y.tolist())
    assert counts.total() == 10_000
    expected_proportion = 0.2  # 1 / 5
    # we should get roughly an even distribution of classes, even though the input data is imbalanced
    for i, count in counts.items():
        assert abs(count / counts.total() - expected_proportion) < 0.02


def test_train_multiclass_probe_scores_highly_on_fully_separable_datasets():
    class1_center = 10 * torch.randn(64)
    class2_center = 10 * torch.randn(64)
    class3_center = 10 * torch.randn(64)

    class1_xs = class1_center + torch.randn(128, 64)
    class2_xs = class2_center + torch.randn(32, 64)
    class3_xs = class3_center + torch.randn(200, 64)

    x = torch.cat([class1_xs, class2_xs, class3_xs])
    y = torch.cat(
        [
            0 * torch.ones(len(class1_xs)),  # 0
            1 * torch.ones(len(class2_xs)),  # 1
            2 * torch.ones(len(class3_xs)),  # 2
        ]
    ).long()

    # shuffle x and y in the same way
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    x_train = x[:300]
    y_train = y[:300]

    x_test = x[300:]
    y_test = y[300:]

    probe = train_multiclass_probe(
        x_train, y_train, num_classes=3, num_epochs=5, batch_size=32
    )

    train_preds = probe(x_train).argmax(dim=1)
    test_preds = probe(x_test).argmax(dim=1)

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    assert train_acc > 0.98
    assert test_acc > 0.98


def test_train_multiclass_probe_scores_similarly_to_sklearn_on_noisy_datasets():
    class1_center = 0.1 * torch.randn(64)
    class2_center = 0.1 * torch.randn(64)
    class3_center = 0.1 * torch.randn(64)

    class1_xs = class1_center + torch.randn(150, 64)
    class2_xs = class2_center + torch.randn(60, 64)
    class3_xs = class3_center + torch.randn(250, 64)

    x = torch.cat([class1_xs, class2_xs, class3_xs])
    y = torch.cat(
        [
            0 * torch.ones(len(class1_xs)),  # 0
            1 * torch.ones(len(class2_xs)),  # 1
            2 * torch.ones(len(class3_xs)),  # 2
        ]
    ).long()

    # shuffle x and y in the same way
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    x_train = x[:300]
    y_train = y[:300]

    x_test = x[300:]
    y_test = y[300:]

    probe = train_multiclass_probe(
        x_train,
        y_train,
        num_classes=3,
        num_epochs=200,
        batch_size=16,
        weight_decay=1e-7,
    )
    sk_probe = LogisticRegression(max_iter=200, class_weight="balanced").fit(
        x_train.numpy(), y_train.numpy()
    )

    test_preds = probe(x_test).argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean().item()

    sk_test_preds = torch.tensor(sk_probe.predict(x_test.numpy()))
    sk_test_acc = (sk_test_preds == y_test).float().mean().item()

    assert sk_test_acc - test_acc < 0.05
