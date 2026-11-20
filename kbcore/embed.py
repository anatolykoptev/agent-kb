"""Local bge-m3 int8 embeddings through onnxruntime. CPU only, in process, no network after the first download
(which runs once by itself when the model is missing from the cache; KB_NO_AUTOFETCH=1 turns that off).

Adapted from zilliztech/memsearch (MIT, commit 3e296a3, src/memsearch/embeddings/onnx.py);
see LICENSE-memsearch. Changes: synchronous (no asyncio), explicit CPUExecutionProvider and a
bounded thread count (never Metal / CoreML / MPS), length-sorted batching with an input cap,
the model is loaded from the local cache only (a missing model is downloaded once, then the lookup is
retried; if that fails it is an error), and the dense output is selected by name.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Sequence

MODEL = "gpahal/bge-m3-onnx-int8"  # MIT; int8 export of BAAI/bge-m3 (MIT), 1024-d dense, 8192-token input
DIM = 1024
MAX_TOKENS = 512  # chunks are <= ~1500 chars; cap keeps the attention cost bounded
THREADS = 6  # intra-op threads per session: bit-identical vectors at any count (checked at 4/6/8/12)
BATCH = 8
WORKERS = 2  # sessions that run batches concurrently in bulk embedding (one session's intra-op threads stop scaling near 8)
BULK_MIN = 32  # fewer texts than this (a query, a small edited file) never opens a second session


class ModelMissing(RuntimeError):
    pass


def fetch_model(model: str = MODEL) -> str:
    """The only network call the tool makes: download of a public model (`kb fetch-model`, or once by itself from OnnxEmbedder)."""
    from huggingface_hub import hf_hub_download

    for f in ("tokenizer.json", "model_quantized.onnx"):
        path = hf_hub_download(model, f)
    return os.path.dirname(path)


class OnnxEmbedder:
    """Construction returns once the files are located and the ONNX session is loading on a worker thread (ORT drops the
    GIL while it reads and prepares the 570 MB graph). The tokenizer parse (~0.2 s, holds the GIL) runs on the caller's
    thread at the first embed, overlapping the session load."""

    def __init__(self, model: str = MODEL, threads: int = THREADS, batch: int = BATCH, workers: int = WORKERS):
        from concurrent.futures import ThreadPoolExecutor

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        def locate() -> tuple[str, str]:
            return (hf_hub_download(model, "tokenizer.json", local_files_only=True),
                    hf_hub_download(model, "model_quantized.onnx", local_files_only=True))

        try:
            self._tok_path, self._model_path = locate()
        except Exception as exc:  # LocalEntryNotFoundError and friends
            missing = ModelMissing(f"model {model} not in the local cache; run: kb fetch-model")
            if os.environ.get("KB_NO_AUTOFETCH") == "1":
                raise missing from exc
            print("kb: embedding model missing from the local cache, downloading it once (~570 MB)...", file=sys.stderr, flush=True)
            try:  # one attempt, no loop: download, then look again
                fetch_model(model)
                self._tok_path, self._model_path = locate()
            except Exception as exc2:
                raise missing from exc2

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self._ort, self._so = ort, so
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kb-onnx")
        self._sess_f = self._pool.submit(self._open)
        self._sess = None
        self._extra: list = []  # sessions 2..workers, opened on the first bulk embed
        self._tok = None
        self._tok_lock = threading.Lock()
        self.model = model
        self.batch = batch
        self.workers = workers

    def _open(self):
        return self._ort.InferenceSession(self._model_path, self._so, providers=["CPUExecutionProvider"])

    def _ready(self) -> None:
        if self._sess is not None:
            return
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(self._tok_path)
        tok.enable_padding(pad_id=1, pad_token="<pad>")
        tok.enable_truncation(max_length=MAX_TOKENS)
        sess = self._sess_f.result()
        names = [o.name for o in sess.get_outputs()]
        self._out = "dense_vecs" if "dense_vecs" in names else names[0]
        self._needs_tt = "token_type_ids" in {i.name for i in sess.get_inputs()}
        self._tok, self._sess = tok, sess
        self._pool.shutdown(wait=False)

    def _sessions(self, n: int) -> list:
        """The first n sessions (opening the missing ones in parallel; each holds its own ~1.3 GB of weights and arenas)."""
        self._ready()
        missing = n - 1 - len(self._extra)
        if missing > 0:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=missing) as pool:
                self._extra += list(pool.map(lambda _: self._open(), range(missing)))
        return [self._sess, *self._extra[: n - 1]]

    def _encode(self, texts: Sequence[str], sess=None) -> list[list[float]]:
        import numpy as np

        self._ready()
        sess = sess or self._sess
        with self._tok_lock:  # one shared Tokenizer; its padding/truncation state is not for concurrent use
            enc = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if self._needs_tt:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = sess.run([self._out], feed)[0]
        if out.ndim == 3:  # last_hidden_state -> CLS pooling
            out = out[:, 0, :]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / norms).tolist()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed in length-sorted batches (padding waste dominates otherwise); order preserved.

        Batch composition is fixed by the sort alone, and a vector depends on its batch (the int8 model quantises
        activations dynamically per batch tensor), so running the same batches on several sessions returns exactly the
        vectors one session would."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batches = [order[s : s + self.batch] for s in range(0, len(order), self.batch)]
        res: list[list[float]] = [[] for _ in texts]
        w = max(1, min(self.workers, len(batches))) if len(texts) >= BULK_MIN else 1
        sessions = self._sessions(w)

        def run(k: int) -> None:
            for bi in range(k, len(batches), w):  # round-robin over the length-sorted batches balances the workers
                idx = batches[bi]
                for i, v in zip(idx, self._encode([texts[i] for i in idx], sessions[k])):
                    res[i] = v

        if w == 1:
            run(0)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=w, thread_name_prefix="kb-embed") as pool:
                list(pool.map(run, range(w)))  # list() re-raises a worker's exception
        return res
