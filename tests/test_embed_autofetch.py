"""A model missing from the HF cache is downloaded once and the lookup retried; KB_NO_AUTOFETCH=1 restores the hard error."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("onnxruntime")
hub = pytest.importorskip("huggingface_hub")

import onnxruntime  # noqa: E402

from kbcore import embed as kbembed  # noqa: E402


@pytest.fixture
def stubbed(monkeypatch):
    """hf_hub_download fails (local-only) until fetch_model has run; InferenceSession is a no-op."""
    state = {"fetched": 0}

    def fake_download(model, filename, **kw):
        if kw.get("local_files_only") and not state["fetched"]:
            raise hub.errors.LocalEntryNotFoundError("not cached")
        return f"/fake/{filename}"

    def fake_fetch(model=kbembed.MODEL):
        state["fetched"] += 1
        return "/fake"

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(kbembed, "fetch_model", fake_fetch)
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: object())
    return state


def test_missing_model_is_fetched_once_then_embedder_is_built(stubbed, monkeypatch, capsys):
    monkeypatch.delenv("KB_NO_AUTOFETCH", raising=False)
    e = kbembed.OnnxEmbedder()
    assert stubbed["fetched"] == 1
    assert e._model_path == "/fake/model_quantized.onnx"
    assert "downloading it once" in capsys.readouterr().err


def test_opt_out_raises_model_missing_without_fetching(stubbed, monkeypatch):
    monkeypatch.setenv("KB_NO_AUTOFETCH", "1")
    with pytest.raises(kbembed.ModelMissing):
        kbembed.OnnxEmbedder()
    assert stubbed["fetched"] == 0


def test_failed_fetch_still_raises_model_missing(stubbed, monkeypatch):
    monkeypatch.delenv("KB_NO_AUTOFETCH", raising=False)

    def boom(model=kbembed.MODEL):
        stubbed["fetched"] += 1
        raise OSError("offline")

    monkeypatch.setattr(kbembed, "fetch_model", boom)
    with pytest.raises(kbembed.ModelMissing):
        kbembed.OnnxEmbedder()
    assert stubbed["fetched"] == 1
