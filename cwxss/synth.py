"""Synthesise CW audio with the impairments that make real signals hard.

This is the foundation of the whole project. Perfect labels are free -- we know
exactly what was sent -- so it gives us training data without a single hour of
hand-labelling, and a test harness that can score any decoder against ground
truth at a stated SNR and speed.

The point is not to make pretty CW. Clean machine-sent CW at 20 dB SNR is
already solved by a twenty-line threshold detector. Everything interesting is in
what is modelled here: an operator whose timing wanders, a signal that fades,
another station 200 Hz away, and noise that buries all of it.
"""
import numpy as np

from morse import DIT, dit_seconds, to_units

DEFAULT_RATE = 8000          # CW lives near 600 Hz; 8 kHz is ample


class Fist:
    """How a human sends, as opposed to a machine.

    A keyer produces exact 1:3 ratios. A hand on a straight key does not, and
    the ways it goes wrong are consistent enough to model: elements drift in
    length, dahs run short or long, and the gaps between characters stretch
    while the operator thinks. Decoders that assume exact ratios fail on real
    operators long before they fail on noise.
    """

    @classmethod
    def bug(cls, rng=None):
        """A semi-automatic key: the dits are machine-made and identical, the
        dahs are formed by hand and are not. Unmistakable on the air, and a
        pattern no decoder that assumes a fixed ratio will handle."""
        return cls(jitter=0.03, dah_ratio=2.4, char_gap_scale=1.2,
                   weight=1.1, dah_jitter=0.28, rng=rng)

    def __init__(self, jitter=0.0, dah_ratio=3.0, char_gap_scale=1.0,
                 word_gap_scale=1.0, weight=1.0, dah_jitter=None, rng=None):
        self.jitter = jitter                  # per-element random variation
        self.dah_ratio = dah_ratio            # 3.0 is textbook; hands vary
        self.char_gap_scale = char_gap_scale  # >1 means a hesitant sender
        self.word_gap_scale = word_gap_scale
        self.weight = weight                  # key-down bias: >1 is "heavy"
        self.dah_jitter = jitter if dah_jitter is None else dah_jitter
        self.rng = rng or np.random.default_rng()

    @classmethod
    def keyer(cls, rng=None):
        """An electronic keyer: exact, the easy case."""
        return cls(jitter=0.0, rng=rng)

    @classmethod
    def good_op(cls, rng=None):
        return cls(jitter=0.06, dah_ratio=3.0, char_gap_scale=1.05,
                   weight=1.02, rng=rng)

    @classmethod
    def rough_op(cls, rng=None):
        """A tired hand on a straight key: the case that breaks decoders."""
        return cls(jitter=0.22, dah_ratio=2.6, char_gap_scale=1.45,
                   word_gap_scale=1.3, weight=1.15, rng=rng)

    @classmethod
    def on_air(cls, rng=None):
        """A hand as rough as the ones actually on the band.

        Measured on off-air recordings, the spread of the key-down lengths runs
        0.248 to 0.412. Every fist above tops out at 0.244, so the roughest
        sending this synthesiser could produce was the *gentlest* thing the
        station had so far recorded. Noise does not explain the difference: the
        same fist measured from 30 dB down to 6 dB moves by about 0.02.

        That gap matters because it is where both decoders fall apart. Across
        0.24 to 0.41 the model drops from 95% to 38% and its lead over the
        timing rules goes from seventeen points to two.

        This is the modal case on the air, not a tail, which is what separates
        it from the Farnsworth widening that was tried and reverted: that one
        spent capacity on beginners' practice tapes, which are rare.
        """
        r = rng if rng is not None else _rng()
        return cls(jitter=float(r.uniform(0.24, 0.50)),
                   dah_ratio=float(r.uniform(2.4, 3.4)),
                   char_gap_scale=float(r.uniform(1.1, 1.8)),
                   word_gap_scale=float(r.uniform(1.0, 1.6)),
                   weight=float(r.uniform(0.9, 1.3)), rng=r)

    def scale(self, on, units):
        """Length of one element, in dit units, as this operator would send it."""
        u = float(units)
        if on and units == DIT * 3:
            u = self.dah_ratio
        if on:
            u *= self.weight
        elif units >= 7:
            u *= self.word_gap_scale
        elif units >= 3:
            u *= self.char_gap_scale
        j = self.dah_jitter if (on and units == DIT * 3) else self.jitter
        if j:
            u *= float(self.rng.normal(1.0, j))
        return max(u, 0.15)          # a hand never sends a truly zero element


def keyed_envelope(text, wpm, fist=None, rate=DEFAULT_RATE, rise_ms=5.0,
                   lead_s=0.3, tail_s=0.3, eff_wpm=None):
    """The key-down envelope over time, 0..1.

    The rise and fall are a raised cosine. Real transmitters shape their keying
    this way; switching a carrier on instantly splatters clicks across the band,
    and a decoder trained on hard-switched edges would be learning an artefact
    that no real signal has.
    """
    fist = fist or Fist.keyer()
    dit_s = dit_seconds(wpm)
    # Farnsworth: characters at full speed, the gaps between them stretched so
    # the overall rate is slower. It is how most people are taught now and how a
    # great many operators send -- and a decoder that reads gaps as fixed
    # multiples of a dit puts a word break between every letter.
    gap_stretch = (wpm / float(eff_wpm)) if eff_wpm and eff_wpm < wpm else 1.0
    edge = max(int(rate * rise_ms / 1000.0), 1)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))

    env = [np.zeros(int(rate * lead_s))]
    for on, units in to_units(text):
        scaled = fist.scale(on, units)
        if not on and units >= 3:
            scaled *= gap_stretch
        n = max(int(round(scaled * dit_s * rate)), 1)
        if not on:
            env.append(np.zeros(n))
            continue
        seg = np.ones(n)
        if n > 2 * edge:
            seg[:edge], seg[-edge:] = ramp, ramp[::-1]
        else:                                   # element shorter than the ramps
            half = max(n // 2, 1)
            shape = 0.5 * (1 - np.cos(np.linspace(0, np.pi, half)))
            seg[:half], seg[-half:] = shape, shape[::-1]
        env.append(seg)
    env.append(np.zeros(int(rate * tail_s)))
    return np.concatenate(env)


def qsb(n, rate, period_s=6.0, depth=0.0, rng=None):
    """Slow fading. Depth 0 is a steady signal, 1 fades to nothing."""
    if depth <= 0:
        return np.ones(n)
    rng = rng or np.random.default_rng()
    t = np.arange(n) / rate
    phase = rng.uniform(0, 2 * np.pi)
    slow = 0.5 * (1 + np.sin(2 * np.pi * t / period_s + phase))
    return (1 - depth) + depth * slow


def tone(env, pitch, rate=DEFAULT_RATE, drift_hz=0.0, rng=None):
    """Turn an envelope into an audible signal at a given pitch."""
    n = len(env)
    t = np.arange(n) / rate
    f = np.full(n, float(pitch))
    if drift_hz:
        rng = rng or np.random.default_rng()
        f += np.linspace(0, rng.uniform(-drift_hz, drift_hz), n)
    phase = 2 * np.pi * np.cumsum(f) / rate
    return env * np.sin(phase)


def add_noise(sig, snr_db, rate=DEFAULT_RATE, bandwidth=500.0, rng=None):
    """Add white noise for a target SNR in a stated bandwidth.

    CW signal-to-noise is quoted in a reference bandwidth, because the number is
    meaningless without one -- the same signal is 10 dB "better" measured in
    50 Hz than in 500. Bandwidth is an argument here for that reason.
    """
    rng = rng or np.random.default_rng()
    on = sig[np.abs(sig) > 1e-6]
    p_sig = float(np.mean(on ** 2)) if on.size else 1e-12
    p_noise_band = p_sig / (10 ** (snr_db / 10.0))
    p_noise_full = p_noise_band * (rate / 2.0) / bandwidth
    return sig + rng.normal(0.0, np.sqrt(p_noise_full), len(sig))


def add_qrn(sig, rate, rate_per_s=0.0, strength=3.0, rng=None):
    """Static crashes: short, loud, broadband impulses.

    Thermal noise is the easy kind. What actually ruins reception on 40m in
    July is lightning -- brief bursts many times louder than the signal, which
    a threshold detector reads as key-down.
    """
    if rate_per_s <= 0:
        return sig
    rng = rng or np.random.default_rng()
    out = sig.copy()
    n_crashes = int(rng.poisson(rate_per_s * len(sig) / rate))
    peak = float(np.max(np.abs(sig))) or 1.0
    for _ in range(n_crashes):
        at = int(rng.integers(0, max(len(sig) - 1, 1)))
        width = int(rng.integers(int(0.002 * rate), int(0.02 * rate) + 1))
        env = np.exp(-np.linspace(0, 6, width))
        burst = rng.normal(0, peak * strength, width) * env
        end = min(at + width, len(out))
        out[at:end] += burst[:end - at]
    return out


def render(text, wpm=20, pitch=600, snr_db=20, fist=None, rate=DEFAULT_RATE,
           qsb_depth=0.0, qrm=None, drift_hz=0.0, seed=None,
           eff_wpm=None, qrn_per_s=0.0):
    """A complete synthetic signal, with everything applied.

    `qrm` is (text, wpm, pitch, level) for a competing station -- the case a
    narrow filter cannot always solve, because the other operator may be only a
    few tens of hertz away.
    """
    rng = np.random.default_rng(seed)
    fist = fist or Fist.keyer(rng)
    env = keyed_envelope(text, wpm, fist, rate, eff_wpm=eff_wpm)
    sig = tone(env, pitch, rate, drift_hz, rng)
    sig *= qsb(len(sig), rate, depth=qsb_depth, rng=rng)

    if qrm:
        q_text, q_wpm, q_pitch, q_level = qrm
        q_env = keyed_envelope(q_text, q_wpm, Fist.good_op(rng), rate)
        q_sig = tone(q_env, q_pitch, rate, 0.0, rng)
        if len(q_sig) < len(sig):
            q_sig = np.pad(q_sig, (0, len(sig) - len(q_sig)))
        sig = sig + q_level * q_sig[:len(sig)]

    sig = add_noise(sig, snr_db, rate, rng=rng)
    sig = add_qrn(sig, rate, qrn_per_s, rng=rng)
    peak = np.max(np.abs(sig)) or 1.0
    return (sig / peak * 0.9).astype(np.float32)
