"""HTTP layer over a real (fp32) engine: streaming and non-streaming agree, chat works,
errors come back as HTTP errors, metrics reflect the pool."""

import json

import pytest
from fastapi.testclient import TestClient
from tokenizers import Tokenizer

from nanoserve.server import Detokenizer, build_app, qwen_chat_prompt

from .conftest import gpu


@pytest.fixture(scope="module")
def tok():
    # Only the tokenizer file, so the CPU-only CI job does not pull 1 GB of weights.
    from huggingface_hub import hf_hub_download

    from nanoserve.model import MODEL_ID
    return Tokenizer.from_file(hf_hub_download(MODEL_ID, "tokenizer.json"))


@pytest.fixture
def client(make_engine, tok):
    app = build_app(make_engine(max_model_len=512), tok, "qwen-test")
    with TestClient(app) as c:
        yield c


def _sse(resp) -> list[dict]:
    events = [line[6:] for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


@gpu
def test_stream_equals_non_stream(client):
    body = {"prompt": "The capital of France is", "max_tokens": 20, "ignore_eos": True}
    full = client.post("/v1/completions", json=body).json()
    assert full["usage"]["completion_tokens"] == 20
    with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as r:
        chunks = _sse(r)
    assert "".join(c["choices"][0]["text"] for c in chunks) == full["choices"][0]["text"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


@gpu
def test_concurrent_requests_are_batched(client):
    import concurrent.futures as cf
    body = {"prompt": "Count: one, two,", "max_tokens": 12, "ignore_eos": True}
    with cf.ThreadPoolExecutor(6) as ex:
        outs = list(ex.map(lambda _: client.post("/v1/completions", json=body).json(), range(6)))
    assert len({o["choices"][0]["text"] for o in outs}) == 1  # greedy: same answer for all
    assert client.get("/metrics").json()["kv_blocks_free"] == client.get("/metrics").json()["kv_blocks_total"]


@gpu
def test_chat_endpoint(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Reply with the single word: yes"}], "max_tokens": 8}).json()
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert r["choices"][0]["message"]["content"].strip()


@gpu
def test_too_long_is_a_400(client):
    r = client.post("/v1/completions", json={"prompt": list(range(100, 600)), "max_tokens": 100})
    assert r.status_code == 400


def test_chat_template_matches_qwen():
    p = qwen_chat_prompt([{"role": "user", "content": "hi"}])
    assert p.startswith("<|im_start|>system\n")
    assert p.endswith("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")


def test_detokenizer_holds_back_partial_utf8(tok):
    ids = tok.encode("café 🙂 done").ids
    d = Detokenizer(tok)
    pieces = [d.push([i]) for i in ids[:-1]] + [d.push([ids[-1]], final=True)]
    assert "".join(pieces) == "café 🙂 done"
    assert all("�" not in p for p in pieces)
