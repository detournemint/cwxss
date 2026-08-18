"""Record real CW off the air, with whatever we know about it.

Live audio is where a decoder trained on synthetic data meets reality, and the
answer to "does the simulation resemble the air" cannot be found in simulation.

The catch is labels. Off-air CW has none: we do not know what was sent, so it
cannot be used for supervised training directly. Three things it can do, in
increasing order of value:

  1. Evaluation. Decode it, read the result, and see where we are wrong. Needs
     no labels for the parts a human checks by eye.
  2. Domain matching. Measure the noise, the fading, the AGC, the filter shape
     and the distribution of speeds actually on the band, then make the
     synthesiser produce that. This needs no labels at all and improves every
     future model.
  3. Labelled real audio, which does exist in two forms: our own transmissions,
     where we know precisely what was sent because we sent it, and code practice
     broadcasts whose text is published.

Recording shells out to arecord rather than binding an audio library, because
the target is a cheap laptop and one less dependency is one less thing to fail
in a field.
"""
import asyncio
import json
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path.home() / "cw-captures"


def list_devices():
    """Capture devices, as ALSA names arecord will accept."""
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return []
    devs = []
    for line in out.splitlines():
        if not line.startswith("card "):
            continue
        try:
            card = int(line.split("card ")[1].split(":")[0])
            dev = int(line.split("device ")[1].split(":")[0])
            name = line.split("[")[1].split("]")[0]
        except (IndexError, ValueError):
            continue
        devs.append({"alsa": f"plughw:{card},{dev}", "name": name,
                     "card": card, "device": dev})
    return devs


def pick_device(devices=None, prefer=("CODEC", "USB Audio")):
    """The radio's codec, not the webcam.

    A laptop's first capture device is usually its own microphone or a camera,
    and recording that produces a beautifully clean file of nothing at all.
    """
    devices = devices if devices is not None else list_devices()
    for want in prefer:
        for d in devices:
            if want.lower() in d["name"].lower():
                return d
    return devices[0] if devices else None


async def record(seconds, path, device=None, rate=8000):
    """Record mono audio to a WAV file. Returns the path, or None."""
    dev = device or (pick_device() or {}).get("alsa")
    if not dev:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "arecord", "-D", dev, "-f", "S16_LE", "-r", str(rate), "-c", "1",
        "-d", str(int(seconds)), "-t", "wav", str(path),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError((err or b"").decode(errors="replace")[:200])
    return path


def read_wav(path):
    """WAV to float32 samples and its sample rate."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
        width, chans = w.getsampwidth(), w.getnchannels()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    a = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if chans > 1:
        a = a.reshape(-1, chans).mean(axis=1)
    return a / float(np.iinfo(dtype).max), rate


def write_sidecar(path, **meta):
    """Everything known about a capture, beside it.

    A WAV on its own is unusable six months later. Frequency, mode, speed, who
    was sending and -- when we know it -- the true text are what make a
    recording worth keeping.
    """
    meta.setdefault("recorded", datetime.now(timezone.utc).isoformat())
    side = Path(path).with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2, default=str))
    return side


def measure(audio, rate):
    """What this recording is actually like, for tuning the synthesiser.

    Comparing these numbers between real captures and generated clips is how we
    find out whether the training data resembles the air, rather than assuming.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dsp
    pitch, sharp = dsp.find_pitch(audio, rate)
    if not pitch:
        return {"pitch": None}
    env = dsp.envelope(audio, pitch, rate)
    on = env > np.percentile(env, 60)
    duty = float(on.mean())
    return {
        "pitch_hz": round(pitch, 1),
        "tone_peak_db": round(sharp, 1),
        "envelope_snr_db": round(dsp.snr_estimate(env), 1),
        "duty_cycle": round(duty, 3),
        "seconds": round(len(audio) / rate, 1),
    }
