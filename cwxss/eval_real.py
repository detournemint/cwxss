"""Score the decoders against real recordings with known transcripts.

Synthetic tests measure whether a decoder can read what we generated, which is a
question we set ourselves. This measures whether it can read text it has never
seen, sent by someone else, at speeds we did not choose.

    python3 cwxss/eval_real.py --model models/cw.onnx
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classic, dsp, guess, neural, score      # noqa: E402


def load(path):
    import wave
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def usable(text):
    """Is this transcript actually text?

    One ARRL download arrived as binary and scored the decoder at 8.7% against
    it, which read as a catastrophic regression and was nothing of the kind --
    the decode was "PL259 FORMAT CONNECTOR", perfectly good copy. A benchmark
    that silently accepts a corrupt reference is worse than no benchmark: it
    produces confident numbers that are wrong.
    """
    if not text or len(text) < 80:
        return False
    printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
    if printable / len(text) < 0.95:
        return False
    letters = sum(1 for c in text.upper() if c.isalpha())
    return letters / len(text) > 0.4


def clean(text):
    """Normalise a transcript for comparison.

    The published text carries control characters standing for prosigns, and
    the announcement at the start is not part of the practice text.
    """
    t = re.sub(r"<[^>]*>", " ", text.upper())
    t = re.sub(r"[^A-Z0-9 .,?/=+-]", " ", t)
    return " ".join(t.split())


def best_offset(truth_words, got, window=40):
    """Where in the transcript this decode belongs, and how well it matches."""
    best, at = 0.0, 0
    span = max(len(got.split()), 4)
    for i in range(0, max(len(truth_words) - span, 1)):
        cand = " ".join(truth_words[i:i + span + 4])[:len(got) + 24]
        a = score.accuracy(cand, got)
        if a > best:
            best, at = a, i
    return best, at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/arrl")
    ap.add_argument("--model", default="models/cw.onnx")
    ap.add_argument("--seconds", type=int, default=45)
    ap.add_argument("--skip", type=int, default=90, help="skip the announcement")
    a = ap.parse_args()

    net = neural.NeuralDecoder(a.model)
    print(f"  model: {'loaded' if net.available else net.error}\n")
    print("  " + "file".ljust(20) + "classic   neural   guess")
    print("  " + "-" * 48)
    rows = []
    for wav in sorted(Path(a.dir).glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        raw = txt.read_text(errors="ignore")
        if not usable(raw):
            print(f"  {wav.stem.ljust(20)}skipped: transcript is not readable text")
            continue
        audio, rate = load(wav)
        truth = clean(raw)
        words = truth.split()
        seg = audio[a.skip * rate:(a.skip + a.seconds) * rate]
        if seg.size < rate:
            continue
        pitch, _ = dsp.find_pitch(seg, rate)
        if not pitch:
            continue
        env = dsp.envelope(seg, pitch, rate, bandwidth=dsp.cw_bandwidth(20))
        norm, _, _ = dsp.normalise(env)
        c_text, info = classic.decode(norm)
        n_text = net.decode(norm) if net.available else ""
        g_text = " ".join(guess.repair(c_text)[0])
        c, _ = best_offset(words, c_text)
        n, _ = best_offset(words, n_text) if n_text else (0.0, 0)
        g, _ = best_offset(words, g_text)
        rows.append((wav.stem, c, n, g, info.get("wpm")))
        print(f"  {wav.stem.ljust(20)}{c*100:6.1f}%  {n*100:6.1f}%  {g*100:6.1f}%"
              f"   ({info.get('wpm')} wpm)")
    if rows:
        print("  " + "-" * 48)
        print(f"  {'mean'.ljust(20)}"
              f"{np.mean([r[1] for r in rows])*100:6.1f}%  "
              f"{np.mean([r[2] for r in rows])*100:6.1f}%  "
              f"{np.mean([r[3] for r in rows])*100:6.1f}%")


if __name__ == "__main__":
    main()
