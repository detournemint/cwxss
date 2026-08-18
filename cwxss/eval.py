"""Score decoders against known text, under stated conditions.

The same clips go to every decoder, so the comparison is like for like. The
conditions are named and fixed rather than random, because "it got better" is
only meaningful if the test did not move.

    python3 cwxss/eval.py                 # classic only
    python3 cwxss/eval.py --model models/cw.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classic, dsp, hamtext, score, synth      # noqa: E402

# Each case is a condition an operator actually meets, named for what it is.
CASES = [
    ("clean keyer 20wpm",      dict(wpm=20, snr_db=25)),
    ("clean keyer 30wpm",      dict(wpm=30, snr_db=25)),
    ("good op 18wpm",          dict(wpm=18, snr_db=20, fist="good")),
    ("rough fist 18wpm",       dict(wpm=18, snr_db=20, fist="rough")),
    ("bug key 20wpm",          dict(wpm=20, snr_db=20, fist="bug")),
    ("Farnsworth 20/12",       dict(wpm=20, snr_db=20, eff_wpm=12)),
    ("weak 6 dB",              dict(wpm=20, snr_db=6)),
    ("weak 3 dB",              dict(wpm=20, snr_db=3)),
    ("QSB deep fade",          dict(wpm=20, snr_db=15, qsb_depth=0.8)),
    ("static crashes",         dict(wpm=20, snr_db=15, qrn_per_s=3.0)),
    ("QRM 150 Hz away",        dict(wpm=20, snr_db=15, qrm=True)),
    ("rough + weak + QRN",     dict(wpm=18, snr_db=6, fist="rough", qrn_per_s=2.0)),
]
FISTS = {"good": synth.Fist.good_op, "rough": synth.Fist.rough_op,
         "bug": synth.Fist.bug, None: synth.Fist.keyer}


def clip(text, rng, **kw):
    kw = dict(kw)
    fist = FISTS[kw.pop("fist", None)](rng)
    if kw.pop("qrm", False):
        kw["qrm"] = (hamtext.sample(rng), 22.0, 750.0, 0.8)
    return synth.render(text, pitch=600, fist=fist,
                        seed=int(rng.integers(0, 2 ** 31)), **kw)


def run(decoders, trials=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, kw in CASES:
        got = {k: [] for k in decoders}
        for _ in range(trials):
            text = hamtext.sample(rng)
            audio = clip(text, rng, **kw)
            env, _, _ = dsp.normalise(dsp.envelope(audio, 600))
            for k, fn in decoders.items():
                got[k].append(score.accuracy(text, fn(env)))
        rows.append((name, {k: float(np.mean(v)) * 100 for k, v in got.items()}))
    return rows


def report(rows, names):
    head = "  " + "condition".ljust(23) + "".join(n.rjust(11) for n in names)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, res in rows:
        line = "  " + name.ljust(23)
        for n in names:
            line += f"{res[n]:9.1f}%"
        if len(names) == 2:
            d = res[names[1]] - res[names[0]]
            line += f"   {d:+6.1f}"
        print(line)
    print("  " + "-" * (len(head) - 2))
    avg = {n: float(np.mean([r[1][n] for r in rows])) for n in names}
    line = "  " + "mean".ljust(23) + "".join(f"{avg[n]:9.1f}%" for n in names)
    if len(names) == 2:
        line += f"   {avg[names[1]] - avg[names[0]]:+6.1f}"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--trials", type=int, default=8)
    a = ap.parse_args()

    decoders = {"classic": lambda env: classic.decode(env)[0]}
    names = ["classic"]
    if a.model and Path(a.model).exists():
        import torch
        import model as M
        net = M.CWNet()
        net.load_state_dict(torch.load(a.model, map_location="cpu")["state"])
        net.eval()

        def neural(env):
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(env, dtype=np.float32))[None]
                return M.decode_greedy(net(x)[0])

        decoders["neural"] = neural
        names.append("neural")

    report(run(decoders, a.trials), names)


if __name__ == "__main__":
    main()
