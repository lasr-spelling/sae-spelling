from sae_spelling.util import batchify


def test_batchify_splits_sequences_into_chunks():
    batches = [batch for batch in batchify(list(range(10)), batch_size=3)]
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]