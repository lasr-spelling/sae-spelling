from sae_spelling.baseline import generate_and_score_samples, get_valid_vocab


def test_get_valid_vocab(hf_gemma2_modeltokenizer):
    _, tokenizer = hf_gemma2_modeltokenizer
    vocab_dict = get_valid_vocab(tokenizer)
    assert isinstance(vocab_dict, dict)
    for key, value in vocab_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, list)
        assert all(len(word) == int(key) for word in value)


def test_generate_and_score_samples(hf_gemma2_modeltokenizer):
    model, tokenizer = hf_gemma2_modeltokenizer
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
        assert isinstance(result, dict)
        assert "word_length" in result
        assert isinstance(result["word_length"], str)
        assert "icl_length" in result
        assert isinstance(result["icl_length"], int)
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1
