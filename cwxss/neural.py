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
# Below this the model is guessing. Chosen from measurement: on clips it reads
# correctly the mean confidence sits well above 0.9, and on pure noise it falls
# to roughly a third of that.
MIN_CONFIDENCE = 0.55
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

    def decode(self, env, min_confidence=MIN_CONFIDENCE):
        """Envelope to text, with the characters it is unsure of left out.

        CTC gives a probability for every frame, and it is worth listening to.
        Without this the decoder is equally fluent whether it is reading a
        signal or reading noise: at -8 dB it produced "CQ GDTL DDXR RRI 5W"
        with no indication that every character was invented. An operator can
        act on silence. They cannot act on confident nonsense.
        """
        text, _ = self.decode_scored(env, min_confidence)
        return text

    def decode_scored(self, env, min_confidence=MIN_CONFIDENCE):
        """Text and mean confidence, 0..1."""
        if not self.available or env is None or len(env) < 40:
            return "", 0.0
        x = np.asarray(env, dtype=np.float32)[None]
        try:
            logprobs = self.sess.run(None, {"envelope": x})[0][0]
        except Exception as e:
            self.error = f"{type(e).__name__}"
            return "", 0.0

        best = logprobs.argmax(axis=-1)
        conf = np.exp(logprobs.max(axis=-1))        # the model exports log-probs
        out, scores, prev = [], [], None
        for k, c in zip(best, conf):
            k = int(k)
            if k != prev and k != BLANK:
                out.append(INV.get(k, ""))
                scores.append(float(c))
            prev = k
        if not out:
            return "", 0.0
        mean_conf = float(np.mean(scores))
        # Judge the passage, not each character. Dropping individual low-scoring
        # characters was tried and cost 9 points during a deep fade: confidence
        # falls across the whole passage as the signal goes, and the characters
        # it discards there are often right. What matters is whether the model
        # is reading a signal at all, which is a property of the passage.
        if mean_conf < min_confidence:
            return "", mean_conf
        return "".join(out).strip(), mean_conf
