import pytest
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture
def gpt2_tokenizer(gpt2_model: HookedTransformer):
    return gpt2_model.tokenizer


@pytest.fixture
def gpt2_l4_sae() -> SAE:
    return SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.4.hook_resid_pre", device="cpu"
    )[0]


@pytest.fixture
def gpt2_l5_sae() -> SAE:
    return SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.5.hook_resid_pre", device="cpu"
    )[0]


@pytest.fixture
def hf_gemma2_modeltokenizer():
    model_name = "gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device="cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, device="cpu")
    return model, tokenizer
