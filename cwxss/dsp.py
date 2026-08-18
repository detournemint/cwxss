"""Audio in, envelope out.

Everything a CW decoder needs is in one narrow band: how loud the operator's
tone is, over time. This finds the tone, extracts that envelope, and throws the
rest of the spectrum away -- which is most of the noise.

numpy only, on purpose. This has to run on the station server, which is four
cores with no GPU and no scipy.
"""
import numpy as np

DEFAULT_RATE = 8000
FRAME_RATE = 100          # envelope samples per second: 10 ms resolution
PITCH_LO, PITCH_HI = 250.0, 1500.0     # where operators actually put their tone


def find_pitch(audio, rate=DEFAULT_RATE, lo=PITCH_LO, hi=PITCH_HI):
    """The strongest tone in the CW range, in Hz.

    A CW signal is a single tone switched on and off, so the loudest narrow peak
    in the band is almost always it. Averaging power over the whole clip rather
    than one window means key-up periods dilute the peak but do not move it.
    """
    n = int(2 ** np.floor(np.log2(min(len(audio), rate * 4))))
    if n < 512 or len(audio) < n:
        return None, 0.0          # not enough audio to say anything honest
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    hops = max((len(audio) - n) // (n // 2) + 1, 1)
    for i in range(hops):
        seg = audio[i * (n // 2):i * (n // 2) + n]
        if len(seg) < n:
            break
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return None, 0.0
    if not np.any(acc[band] > 0):
        return None, 0.0
    idx = np.argmax(np.where(band, acc, 0.0))
    peak, floor = acc[idx], np.median(acc[band])
    return float(freqs[idx]), float(10 * np.log10((peak + 1e-20) / (floor + 1e-20)))


def _boxcar(x, n):
    """Moving average. A boxcar is a poor filter in general and the right one
    here: it is cheap, linear phase, and its width sets the bandwidth directly."""
    if n <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, out[0] if len(out) else 0.0), out])


def envelope(audio, pitch, rate=DEFAULT_RATE, bandwidth=200.0,
             frame_rate=FRAME_RATE):
    """Magnitude of the signal in a narrow band around `pitch`, decimated.

    Mixing the tone down to DC and low-passing is the same operation as a
    band-pass filter centred on it, but it costs one multiply per sample and the
    bandwidth is set by one number.
    """
    n = len(audio)
    t = np.arange(n) / rate
    mixed = audio * np.exp(-2j * np.pi * pitch * t)
    taps = max(int(rate / bandwidth), 1)
    base = _boxcar(mixed.real, taps) + 1j * _boxcar(mixed.imag, taps)
    mag = np.abs(base)

    step = max(int(rate / frame_rate), 1)
    usable = (len(mag) // step) * step
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return mag[:usable].reshape(-1, step).max(axis=1).astype(np.float32)


def normalise(env, floor_pct=20, peak_pct=95):
    """Scale an envelope to roughly 0..1 using percentiles.

    The minimum and maximum are the two worst choices: one noise spike sets the
    top and silence sets the bottom. Percentiles ignore both.
    """
    if env.size == 0:
        return env, 0.0, 0.0
    floor = float(np.percentile(env, floor_pct))
    peak = float(np.percentile(env, peak_pct))
    span = max(peak - floor, 1e-9)
    return np.clip((env - floor) / span, 0.0, 2.0).astype(np.float32), floor, peak


def snr_estimate(env):
    """Rough on/off signal-to-noise of a RAW envelope, in dB.

    Not a calibrated measurement -- it is the separation between the loud and
    quiet parts of what we are looking at, which is what actually predicts
    whether a decode will work.

    Must be given the envelope before normalise(), not after. Normalising puts
    the noise floor at zero by construction, so the ratio against it explodes:
    a 6 dB signal cheerfully reported itself as 79 dB that way.
    """
    if env.size < 10:
        return 0.0
    lo = float(np.percentile(env, 20))
    hi = float(np.percentile(env, 95))
    if hi <= 0:
        return 0.0
    # A near-zero floor means a very clean signal, not a broken measurement.
    # Returning 0 dB for it -- as this did on an ARRL practice file, the
    # cleanest audio available -- says the opposite of what is true.
    floor = max(lo, hi * 1e-3)
    return float(min(20 * np.log10(hi / floor), 60.0))
