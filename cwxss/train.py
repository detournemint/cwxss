"""Train the neural decoder on synthetic CW.

There is no dataset on disk. Every clip is generated fresh, so the model never
sees the same example twice and no amount of training can overfit a fixed set.
The impairments are drawn at random per clip: speed, pitch, noise, fading, a
competing station, and how steady the operator's hand is.

    python3 cwxss/train.py --steps 4000 --out models/cw.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp, hamtext, model as M, synth        # noqa: E402


def make_clip(rng, wpm=None, snr=None, fist=None, text=None, pitch=None,
              qsb_depth=None, qrm=None):
    """One (envelope, text) pair with randomised conditions."""
    text = text if text is not None else hamtext.sample(rng)
    # Down to 8 wpm: ARRL practice starts at 5 wpm effective and beginners on
    # the air are slower than the model had ever seen.
    wpm = wpm if wpm is not None else float(rng.uniform(8, 38))
    snr = snr if snr is not None else float(rng.uniform(-2, 28))
    pitch = pitch if pitch is not None else float(rng.uniform(400, 900))
    if fist is None:
        r = rng.random()
        fist = (synth.Fist.keyer(rng) if r < 0.25 else
                synth.Fist.good_op(rng) if r < 0.60 else
                synth.Fist.bug(rng) if r < 0.75 else
                synth.Fist.rough_op(rng))
    # Farnsworth on a third of clips. Without it the model learns that a gap of
    # a given length always means the same thing, which is false for most of the
    # operators it will actually meet.
    eff = None
    if rng.random() < 0.40:
        # Up to a 3:1 ratio. ARRL sends its 5 wpm practice as 15 wpm characters
        # with enormous gaps, and a model that has only seen 2:1 reads the gaps
        # as word breaks -- 35% on a file that is otherwise pristine. Beginners
        # are taught this way and a great many of them send this way.
        eff = wpm / float(rng.uniform(1.05, 3.2))
    # Static crashes on a quarter, because a threshold detector reads a
    # lightning burst as key-down and the model needs to learn not to.
    qrn = float(rng.uniform(0.5, 4.0)) if rng.random() < 0.25 else 0.0
    if qsb_depth is None:
        qsb_depth = float(rng.uniform(0, 0.8)) if rng.random() < 0.35 else 0.0
    if qrm is None and rng.random() < 0.25:
        # a competing signal close enough that the filter cannot fully remove it
        offset = float(rng.choice([-1, 1]) * rng.uniform(60, 300))
        qrm = (hamtext.sample(rng), float(rng.uniform(12, 30)),
               max(300.0, pitch + offset), float(rng.uniform(0.3, 0.9)))

    audio = synth.render(text, wpm=wpm, pitch=pitch, snr_db=snr, fist=fist,
                         qsb_depth=qsb_depth, qrm=qrm, eff_wpm=eff,
                         qrn_per_s=qrn, seed=int(rng.integers(0, 2**31)))
    # The decoder is given the pitch it should listen on, as it would be live:
    # the front end finds the tone, the model reads the envelope.
    env = dsp.envelope(audio, pitch)
    env, _, _ = dsp.normalise(env)
    return env, text


def batch(rng, size, max_frames=1400):
    envs, texts = [], []
    while len(envs) < size:
        e, t = make_clip(rng)
        if len(e) > max_frames or not t:
            continue
        envs.append(e)
        texts.append(t)
    width = max(len(e) for e in envs)
    x = torch.zeros(len(envs), width)
    for i, e in enumerate(envs):
        x[i, :len(e)] = torch.from_numpy(e)
    targets = [M.encode_text(t) for t in texts]
    y = torch.tensor([k for t in targets for k in t], dtype=torch.long)
    in_lens = torch.full((len(envs),), width, dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return x, y, in_lens, tgt_lens, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="models/cw.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    rng = np.random.default_rng(a.seed)

    net = M.CWNet()
    print(f"  {M.parameter_count(net):,} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps)
    ctc = nn.CTCLoss(blank=M.BLANK, zero_infinity=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    net.train()
    t0, run = time.time(), None
    for step in range(1, a.steps + 1):
        x, y, in_lens, tgt_lens, _ = batch(rng, a.batch)
        logits = net(x).transpose(0, 1)              # CTC wants (T, N, C)
        loss = ctc(logits, y, in_lens, tgt_lens)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 50 == 0 or step == 1:
            rate = step / (time.time() - t0)
            print(f"  step {step:>5}/{a.steps}  loss {run:6.3f}  "
                  f"{rate:4.1f} steps/s  eta {(a.steps-step)/rate/60:5.1f} min",
                  flush=True)
        if step % 500 == 0 or step == a.steps:
            torch.save({"state": net.state_dict(), "step": step}, a.out)
    torch.save({"state": net.state_dict(), "step": a.steps}, a.out)
    print(f"  saved {a.out} after {a.steps} steps in "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
