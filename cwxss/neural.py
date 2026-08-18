"""The trained decoder, at runtime.

Loaded from ONNX rather than torch: torch is a training tool and a large
dependency, and a radio in a park does not need it. The exported model is 710 KB
and runs about 1500x real time on one core, so it costs nothing to run beside
the classic decoder rather than instead of it.

Absent onnxruntime or a model file, this reports itself unavailable and the
station carries on with the classic decoder alone.
"""
from pathlib import Path

import numpy as np

ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./?=,-+"
BLANK = 0
INV = {i + 1: c for i, c in enumerate(ALPHABET)}


class NeuralDecoder:
    def __init__(self, path="models/cw.onnx"):
        self.path = Path(path)
        self.sess = None
        self.error = ""
        self._load()

    def _load(self):
        if not self.path.exists():
            self.error = f"no model at {self.path}"
            return
        try:
            import onnxruntime as ort
        except ImportError:
            self.error = "onnxruntime not installed"
            return
        try:
            opts = ort.SessionOptions()
            # One thread: this runs beside everything else on a small machine,
            # and at 1500x real time there is nothing to gain from more.
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(
                str(self.path), opts, providers=["CPUExecutionProvider"])
            self.error = ""
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self):
        return self.sess is not None

    def decode(self, env):
        """Envelope to text. Empty string when unavailable or given nothing."""
        if not self.available or env is None or len(env) < 40:
            return ""
        x = np.asarray(env, dtype=np.float32)[None]
        try:
            logits = self.sess.run(None, {"envelope": x})[0][0]
        except Exception as e:
            self.error = f"{type(e).__name__}"
            return ""
        best = logits.argmax(axis=-1)
        out, prev = [], None
        for k in best:
            k = int(k)
            if k != prev and k != BLANK:
                out.append(INV.get(k, ""))
            prev = k
        return "".join(out).strip()
