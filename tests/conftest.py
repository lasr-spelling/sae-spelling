import pytest
from transformer_lens import HookedTransformer


@pytest.fixture
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")
