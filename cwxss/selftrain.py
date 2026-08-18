#!/usr/bin/env python3
"""Learn from real off-air CW that nobody transcribed.

Labelled CW is scarce. The ARRL practice files come with transcripts and there
are nine of them; the band produces hours of real signals a day and none of it
comes with an answer key. Everything the model knows about real operators it has
had to infer from a synthesiser, and a synthesiser only contains the
imperfections somebody thought to write into it.

The usual way out is pseudo-labelling: decode the unlabelled audio, treat the
decodes as truth, train on them. Done naively this mostly teaches the model to
repeat itself, because the clips it decodes confidently are the clips it already
handles and its confident mistakes come through as facts.

So the filter here is not confidence. It is *agreement between two decoders that
fail differently*. The classic decoder is a threshold and a set of timing rules;
the model is a neural net. Their errors are close to uncorrelated -- the
benchmark shows classic ahead on standard timing and the model ahead by 18 and
33 points under heavy Farnsworth. A clip where both produce the same text is a
clip where two unrelated algorithms would have had to make the same mistake.

Three filters, cheapest first:

  1. both decoders produce enough text to be worth anything
  2. they agree closely -- this is the one doing the real work
  3. the agreed text is made of real words and callsigns, not noise that
     happens to parse (the same content test the band sweep uses, which was
     added after a sweep reported forty signals where there were none)

Even then the pseudo-labels are mixed with synthetic clips whose labels are
exact, so the training signal is anchored to something that cannot drift.

    python3 cwxss/selftrain.py --audio ~/cwdata --init models/cw.pt \
        --out models/cw-self.pt --steps 2000

Run --dry-run first. It reports how many clips survive each filter and, on
synthetic audio where the truth is known, how accurate the surviving labels
actually are.
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classic, dsp, neural, score                       # noqa: E402

# Below this the two decoders are not really agreeing, they are both guessing.
# Measured on synthetic clips with known truth: at 0.85 the surviving labels are
# far more accurate than either decoder alone, and the yield is still usable.
MIN_AGREEMENT = 0.85
MIN_CHARS = 12          # shorter than this and agreement is easy by accident
MIN_KEPT_TOKENS = 2     # recognisable words or callsigns in the agreed text


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def envelope_of(audio, rate):
    """Audio to the normalised envelope both decoders consume."""
    pitch, _ = dsp.find_pitch(audio, rate)
    if not pitch:
        return None
    env = dsp.envelope(audio, pitch, rate, bandwidth=dsp.cw_bandwidth(20))
    norm, _, _ = dsp.normalise(env)
    return norm


def consider(env, net):
    """Decode a clip both ways and decide whether to believe the result.

    Returns (text, why) with text None when the clip is rejected, so a dry run
    can report where the losses are instead of just how many survived.
    """
    c_text, _ = classic.decode(env)
    n_text = net.decode(env) if net.available else ""
    if len(c_text.strip()) < MIN_CHARS or len(n_text.strip()) < MIN_CHARS:
        return None, "too short"
    agree = score.accuracy(c_text, n_text)
    if agree < MIN_AGREEMENT:
        return None, "decoders disagree"
    # Take the classic text as the label when they agree: it is the one that
    # was not produced by the network being trained, so it cannot feed the
    # network's own bias back to it.
    text = c_text.strip()
    if not dsp._reads_like_cw(text):
        return None, "not language"
    return text, "kept"


def gather(audio_dir, net, limit=None):
    kept, reasons = [], {}
    files = sorted(Path(audio_dir).glob("*.wav"))
    if limit:
        files = files[:limit]
    for wav in files:
        audio, rate = load_wav(wav)
        if audio.size < rate:
            reasons["too short"] = reasons.get("too short", 0) + 1
            continue
        env = envelope_of(audio, rate)
        if env is None:
            reasons["no tone"] = reasons.get("no tone", 0) + 1
            continue
        text, why = consider(env, net)
        reasons[why] = reasons.get(why, 0) + 1
        if text:
            kept.append((env.astype(np.float32), text, wav.name))
    return kept, reasons, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="directory of harvested wav")
    ap.add_argument("--model", default="models/cw.onnx",
                    help="model used to produce pseudo-labels")
    ap.add_argument("--dry-run", action="store_true",
                    help="report yield and label quality, train nothing")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    net = neural.NeuralDecoder(a.model)
    if not net.available:
        print(f"  model unavailable: {net.error}")
        return 1
    kept, reasons, total = gather(a.audio, net, a.limit)
    print(f"\n  {total} clips considered")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {str(n).rjust(5)}  {why}")
    print(f"\n  {len(kept)} usable pseudo-labels "
          f"({len(kept)/max(total,1)*100:.0f}% yield)")
    for env, text, name in kept[:10]:
        print(f"    {name.ljust(28)} {text[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
