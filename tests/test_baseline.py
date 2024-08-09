from sae_spelling.baseline import (
    BaselineResult,
    generate_and_score_samples,
    get_valid_vocab,
)


def test_get_valid_vocab(gpt2_tokenizer):
    tokenizer = gpt2_tokenizer
    vocab_dict = get_valid_vocab(tokenizer)
    assert isinstance(vocab_dict, dict)
    for key, value in vocab_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, list)
        assert all(len(word) == int(key) for word in value)


def test_generate_and_score_samples(gpt2_hf_model):
    model, tokenizer = gpt2_hf_model
    model = model.to(
        "cpu"
    )  # sometimes this fails in testing, so trying to enforce this
    vocab_dict = get_valid_vocab(tokenizer)
    generator = generate_and_score_samples(
        model,
        tokenizer,
        vocab_dict,
        samples_per_combo=1,
        max_icl_length=1,
    )
    results = list(generator)
    assert len(results) > 0, "Generator yielded no results"
    for result in results:
        assert isinstance(result, BaselineResult)
        assert isinstance(result.word_length, str)
        assert isinstance(result.icl_length, int)
        assert isinstance(result.accuracy, float)
        assert 0 <= result.accuracy <= 1
