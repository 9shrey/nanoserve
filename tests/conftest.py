"""GPU fixtures. The suite runs in fp32 so that "identical to HF" can be asserted exactly:
in fp16, two correct implementations that sum in a different order can flip a near-tied
argmax and diverge from there, which would make the tests measure noise. fp16 agreement is
measured separately by the benchmark's correctness gate."""

import pytest
import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "List three facts about the moon:\n1.",
    "Repeat after me exactly: the quick brown fox jumps over the lazy dog. The quick brown fox",
    "Translate to French: I would like a cup of coffee, please.",
    "A very long story begins here. " * 12 + "And then",
]
MAX_NEW = 24

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="session")
def model_dir():
    from nanoserve.model import resolve_model_dir
    return resolve_model_dir()


@pytest.fixture(scope="session")
def tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


@pytest.fixture(scope="session")
def prompt_ids(tokenizer):
    return [tokenizer(p).input_ids for p in PROMPTS]


@pytest.fixture(scope="session")
def hf_greedy(model_dir, prompt_ids):
    """Reference greedy continuations from HF generate, one prompt at a time (no padding).

    Qwen's generation_config ships repetition_penalty=1.1, which HF applies even with
    do_sample=False. Pure greedy needs it overridden, or the reference is not greedy."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
    out = []
    with torch.no_grad():
        for ids in prompt_ids:
            g = hf.generate(torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                            min_new_tokens=MAX_NEW, pad_token_id=0,
                            repetition_penalty=1.0, top_k=None, top_p=None, temperature=None)
            out.append(g[0, len(ids):].tolist())
    del hf
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="session")
def model32(model_dir, hf_greedy):  # depends on hf_greedy so HF is freed before this loads
    from nanoserve.model import Qwen2Model
    return Qwen2Model(model_dir, dtype=torch.float32, use_skinny_gemm=False)


@pytest.fixture
def make_engine(model32):
    from nanoserve.engine import EngineConfig, LLMEngine

    def make(**kw):
        kw.setdefault("num_kv_blocks", 256)
        kw.setdefault("dtype", torch.float32)
        return LLMEngine(EngineConfig(**kw), model=model32)
    return make
