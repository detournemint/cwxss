"""Collect real CW overnight, unattended.

The one thing this project cannot generate is genuine off-air audio: real
propagation, real noise, real operators with real fists. It can only be gathered
while a radio is switched on and pointed at a band, which is most of the time
nobody is watching.

So this sweeps a band, finds signals that decode like Morse, parks on the best
one, records it, and writes down everything known about it. Overnight it builds
a corpus that no amount of simulation can substitute for -- and because both
decoders run over each recording, it also produces a running comparison on real
signals rather than synthetic ones.

    python3 cwxss/harvest.py --bands 40,30,20 --minutes 480

It transmits nothing. The radio is only ever tuned and listened to.
"""
import argparse
import asyncio
import json
import socket
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classic, dsp, guess, neural, rbn, score  # noqa: E402

# Where CW actually lives on each band, avoiding the digital segments.
SEGMENTS = {
    "160": (1810000, 1840000), "80": (3500000, 3570000),
    "40": (7000000, 7070000), "30": (10100000, 10130000),
    "20": (14000000, 14070000), "17": (18068000, 18095000),
    "15": (21000000, 21070000), "12": (24890000, 24915000),
    "10": (28000000, 28070000),
}

# 60m is not a band, it is five channels. US amateurs share it with government
# users on a secondary basis and are required to keep the emission centred on
# the channel, so there is no segment to sweep -- stepping across it in 2.5 kHz
# hops would spend almost all its time on spectrum where nobody is permitted to
# transmit. These are the channel centres, which is where CW goes.
#
# It is listed here because the nets that run there are exactly the recordings
# this harvester wants: scheduled, regular, and with a net control operator
# calling the roster, so the callsigns are known in advance.
CHANNELS = {
    "60": (5332000, 5348000, 5358500, 5373000, 5405000),
}
RATE = 8000
# How many of a band's candidates to re-listen to. The fast pass ranks them by
# strength, and going deeper than this costs a minute each for steadily worse
# odds.
CANDIDATES = 4


def rig(cmd, host="127.0.0.1", port=4532, wait=0.35):
    try:
        s = socket.create_connection((host, port), timeout=6)
    except OSError:
        return None
    try:
        s.sendall((cmd + "\n").encode())
        time.sleep(wait)
        return s.recv(8192).decode(errors="replace").strip()
    except OSError:
        return None
    finally:
        s.close()


def record(seconds, device, path):
    import subprocess
    path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["arecord", "-D", device, "-f", "S16_LE", "-r", str(RATE),
                        "-c", "1", "-d", str(int(seconds)), "-t", "wav", str(path)],
                       capture_output=True)
    return path if r.returncode == 0 else None


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def rbn_pass(a, model, log):
    """Record where skimmers say there is something to hear.

    Blind sweeping is a lottery: four seconds per step, back around every two
    and a half minutes, against a station that transmits maybe a third of the
    time. Skimmers listen to every frequency at once and publish the answer, so
    this asks them where to point the radio instead of guessing.
    """
    spots = rbn.active_frequencies(a.call, seconds=a.rbn, log=log)
    if not spots:
        log("  rbn: nothing being spotted by nearby skimmers")
        return 0
    log(f"  rbn: {len(spots)} frequency(s) with a CQ nearby")
    kept = 0
    for sp in spots[:CANDIDATES]:
        dial = int(round(sp["khz"] * 1000))
        log(f"    {sp['khz']:.1f} {sp['dx']} {sp['snr']}dB {sp['wpm']}wpm "
            f"via {sp['spotter']}")
        rig(f"F {dial}")
        rig("M CW 500")
        time.sleep(1.5)
        tmp = Path("/tmp/cwxss-rbn.wav")
        if not record(a.record, a.device, tmp):
            continue
        audio = read_wav(tmp)
        real = dsp.find_cw_signals(audio, RATE, net=model)
        if not real:
            log("      heard nothing here")
            continue
        best = dict(real[0])
        best["dial"] = dial
        best["band"] = ""
        best["rbn_dx"] = sp["dx"]
        out, meta = keep(audio, best, a.out, model)
        kept += 1
        log(f"      kept {out.name} [{real[0].get('read_by')}] "
            f"{real[0]['sample'][:34]!r}")
    return kept


def check_receiver(log, probe_hz=7030000):
    """Is the front end actually switched on?

    A power cycle resets the preamp on some rigs, and on the higher bands that
    leaves the receiver deaf without anything looking wrong: the S-meter reads a
    steady floor, the audio is clean, and nothing decodes. Measured on an
    FT-991A after a power cycle, 20m read -54 dB with the preamp off and -8 dB
    with it on. Forty-six decibels of receiver, silently unused.

    Comparing the noise floor with the preamp off and on says whether an
    antenna is connected as well: with no antenna the preamp lifts almost
    nothing.

    It has to be measured somewhere with noise to lift, which means a low band.
    Atmospheric noise dominates on 40m and the receiver's own floor dominates
    on 15m, so the same working antenna measures very differently depending on
    where the dial happens to be sitting:

        3.5 MHz  +42 dB      14.0 MHz  +33 dB
        7.0 MHz  +45 dB      21.0 MHz   +1 dB
       10.1 MHz  +43 dB      28.0 MHz  +21 dB

    An earlier version measured wherever the previous run had left the radio,
    found it on 15m in the evening, and announced that the antenna was
    disconnected. It was not; there was simply nothing arriving to amplify.
    """
    where = rig("f")
    def floor():
        vals = []
        for _ in range(3):
            v = rig("l STRENGTH")
            try:
                vals.append(int(float(v)))
            except (TypeError, ValueError):
                pass
            time.sleep(0.6)
        return max(vals) if vals else None

    rig(f"F {probe_hz}"); time.sleep(0.8)
    rig("L PREAMP 0"); time.sleep(1.5)
    off = floor()
    rig("L PREAMP 20"); time.sleep(1.5)
    on = floor()
    if where:
        rig(f"F {where}")
    if off is None or on is None:
        log("  receiver check: no S-meter reading")
        return
    lift = on - off
    log(f"  receiver: noise floor {off} dB without preamp, {on} dB with it "
        f"({lift:+d} dB)")
    if lift < 6:
        log(f"  WARNING: at {probe_hz/1e6:.3f} MHz the preamp barely changes "
            "the noise floor. On a low band that means no antenna is "
            "connected.")
    else:
        log("  antenna is connected; leaving the preamp on")


def sweep(band, device, dwell, log, net=None):
    """Step across a band's CW segment, returning what was found where."""
    if band in CHANNELS:
        freqs = list(CHANNELS[band])
    else:
        lo, hi = SEGMENTS[band]
        # A wide filter shows the whole passband at once, so the step can be
        # kilohertz rather than hertz -- roughly one step per filter width.
        freqs = list(range(lo, hi, 2500))
    found = []
    tmp = Path("/tmp/cwxss-sweep.wav")
    for freq in freqs:
        rig(f"F {freq}")
        time.sleep(0.6)
        if not record(dwell, device, tmp):
            continue
        audio = read_wav(tmp)
        # No language test on the fast pass: four seconds is not enough text
        # to judge, so this stage only asks "does anything here key like CW".
        for sig in dsp.find_cw_signals(audio, RATE, net=net,
                                       require_language=False):
            sig["dial"] = freq
            sig["band"] = band + "m"
            found.append(sig)
            log(f"    {freq/1e6:.4f} +{sig['audio_hz']:.0f}Hz  "
                f"snr {sig['snr']:.0f}  {sig['wpm']:.0f} wpm  "
                f"[{sig.get('read_by','?')}]  {sig['sample'][:28]!r}")
    return sorted(found, key=lambda s: -s["snr"])


def keep(audio, meta, out_dir, model):
    """Save a recording with its metadata and what both decoders made of it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) / f"cw-{stamp}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    pitch, _ = dsp.find_pitch(audio, RATE)
    text = nn_text = ""
    if pitch:
        env = dsp.envelope(audio, pitch, RATE,
                           bandwidth=dsp.cw_bandwidth(meta.get("wpm")))
        norm, _, _ = dsp.normalise(env)
        text, info = classic.decode(norm)
        meta["measured_wpm"] = info.get("wpm")
        meta["envelope_snr"] = round(dsp.snr_estimate(env), 1)
        if model.available:
            nn_text = model.decode(norm)
    meta.update({
        "recorded": datetime.now(timezone.utc).isoformat(),
        "pitch_hz": round(pitch, 1) if pitch else None,
        "classic": text[:600], "neural": nn_text[:600],
        "agreement": round(score.accuracy(text, nn_text), 3) if text and nn_text else None,
        "guess": " ".join(guess.repair(text)[0])[:600],
    })
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out, meta


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="40,30,20")
    ap.add_argument("--minutes", type=int, default=480)
    ap.add_argument("--device", default="plughw:2,0")
    ap.add_argument("--out", default=str(Path.home() / "cw-captures"))
    ap.add_argument("--model", default=str(Path.home() / "models/cw.onnx"))
    ap.add_argument("--dwell", type=int, default=4, help="seconds per sweep step")
    ap.add_argument("--record", type=int, default=90, help="seconds per keeper")
    ap.add_argument("--rbn", type=int, default=0, metavar="SECONDS",
                    help="listen to the Reverse Beacon Network for this long "
                         "and record where nearby skimmers hear a CQ, instead "
                         "of sweeping blind")
    ap.add_argument("--call", default="K6XSS",
                    help="callsign used to log in to the RBN feed")
    a = ap.parse_args()

    def log(m):
        print(f"{datetime.now(timezone.utc):%H:%M:%S} {m}", flush=True)

    model = neural.NeuralDecoder(a.model)
    log(f"harvest starting: bands {a.bands}, {a.minutes} min, "
        f"model {'loaded' if model.available else model.error}")
    before = rig("f"), rig("m")
    log(f"  restoring to {before} when done")

    known = set(SEGMENTS) | set(CHANNELS)
    bands = [b.strip() for b in a.bands.split(",") if b.strip() in known]

    # Probe on the lowest band being swept: that is where atmospheric noise is
    # strongest and a dead antenna is most obvious. In --rbn mode there are no
    # bands to sweep, so fall back to 40m, which always has noise.
    _lows = [SEGMENTS[b][0] for b in bands if b in SEGMENTS] + \
            [min(CHANNELS[b]) for b in bands if b in CHANNELS]
    check_receiver(log, probe_hz=min(_lows) if _lows else 7030000)

    end = time.time() + a.minutes * 60
    kept = 0
    if a.rbn:
        try:
            while time.time() < end:
                kept += rbn_pass(a, model, log)
        except KeyboardInterrupt:
            pass
        finally:
            if before[0]:
                rig(f"F {before[0]}")
            if before[1]:
                parts = before[1].split()
                if len(parts) >= 2:
                    rig(f"M {parts[0]} {parts[1]}")
            log(f"harvest done: {kept} recordings kept")
        return
    try:
        while time.time() < end:
            for band in bands:
                if time.time() >= end:
                    break
                rig("M CW 3000")
                log(f"  sweeping {band}m")
                hits = sweep(band, a.device, a.dwell, log, net=model)
                if not hits:
                    log(f"  {band}m: nothing")
                    continue
                log(f"  {band}m: {len(hits)} candidate(s), listening properly")
                # Re-listen to the best few for long enough that the language
                # test means something. The fast pass deliberately let noise
                # through; this is where it gets thrown out.
                confirmed = None
                for cand in hits[:CANDIDATES]:
                    rig(f"F {cand['dial']}")
                    rig("M CW 500")
                    time.sleep(1)
                    tmp = Path("/tmp/cwxss-keep.wav")
                    if not record(a.record, a.device, tmp):
                        continue
                    audio = read_wav(tmp)
                    real = dsp.find_cw_signals(audio, RATE, net=model)
                    if not real:
                        log(f"    {cand['dial']/1e6:.4f} was not a station")
                        continue
                    confirmed = dict(cand)
                    confirmed.update(real[0])
                    log(f"    {cand['dial']/1e6:.4f} confirmed "
                        f"[{real[0].get('read_by')}] {real[0]['sample'][:30]!r}")
                    break
                if confirmed is None:
                    continue
                best = confirmed
                out, meta = keep(audio, dict(best), a.out, model)
                kept += 1
                log(f"  kept {out.name}: classic {meta.get('classic','')[:44]!r}")
                if meta.get("neural"):
                    log(f"       neural  {meta['neural'][:44]!r} "
                        f"(agreement {meta.get('agreement')})")
    except KeyboardInterrupt:
        pass
    finally:
        # Restore the mode as well as the dial. Sweeping opens the filter to
        # 3000 Hz to find signals, and an earlier version put the frequency
        # back but not the filter, so the live decoder was left listening
        # through six times the bandwidth it wants -- silently, because
        # everything still works, just worse. The log claimed both had been
        # restored, which is how it went unnoticed.
        if before[0]:
            rig(f"F {before[0]}")
        if before[1]:
            parts = before[1].split()
            if len(parts) >= 2:
                rig(f"M {parts[0]} {parts[1]}")
            elif parts:
                rig(f"M {parts[0]} 0")
        now = rig("f"), rig("m")
        ok = "restored" if now == before else f"WANTED {before}, GOT {now}"
        log(f"harvest done: {kept} recordings kept, {ok}")


if __name__ == "__main__":
    asyncio.run(main())
