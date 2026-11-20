"""The real ONNX embedder (needs the cached model; skipped without it): bulk embedding on several sessions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
hub = pytest.importorskip("huggingface_hub")

from kbcore import embed as kbembed  # noqa: E402


def _model_cached() -> bool:
    try:
        hub.hf_hub_download(kbembed.MODEL, "model_quantized.onnx", local_files_only=True)
        hub.hf_hub_download(kbembed.MODEL, "tokenizer.json", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(), reason="bge-m3 int8 not in the local HF cache (run: kb fetch-model)")
def test_two_sessions_return_the_vectors_one_session_returns_in_the_same_order():
    # A vector depends on its batch (dynamic int8 quantisation per batch tensor), so splitting the same batches over
    # sessions must change nothing; a swapped, dropped or reordered batch would store the wrong vector for a chunk
    # and no error would show. 40 distinct lines = 5 batches, past BULK_MIN.
    subjects = ["the scheduler", "a replica", "the build cache", "our parser", "the billing job"]
    verbs = ["retries after a timeout", "drops stale entries", "pins the library version", "logs every failure", "waits for the lock"]
    texts = [f"Note {i}: {subjects[i % 5]} {verbs[(i // 5) % 5]} when the queue holds {i * 7 + 3} items." for i in range(40)]
    assert len(set(texts)) == 40
    one = kbembed.OnnxEmbedder(workers=1, threads=2)
    two = kbembed.OnnxEmbedder(workers=2, threads=3)
    a = np.array(one.embed(texts))
    b = np.array(two.embed(texts))
    assert len(two._extra) == 1, "the second session was never opened: the bulk path did not run"
    assert a.shape == b.shape == (40, kbembed.DIM)
    assert float(np.abs(a - b).max()) < 1e-5
    # independent of the batching code under test: each text embedded alone (batch of one) lies nearest to its own row
    solo = np.array([one.embed([t])[0] for t in texts])
    assert (np.argmax(b @ solo.T, axis=1) == np.arange(40)).all(), "a vector landed on the wrong chunk"
