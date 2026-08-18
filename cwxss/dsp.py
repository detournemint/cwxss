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


def cw_bandwidth(wpm):
    """How much spectrum a CW signal at this speed actually occupies.

    Keying a carrier on and off spreads it either side of the tone by roughly
    four times the element rate. At 20 wpm a dit is 60 ms, so about 17 elements
    a second, so about 70 Hz -- and listening to any more than that is listening
    to noise. Going from 200 Hz to 60 Hz on a real off-air signal was worth
    10 dB, which is the difference between garbage and copy.
    """
    if not wpm or wpm <= 0:
        return 80.0
    elements_per_s = wpm * 50.0 / 60.0        # PARIS: 50 units per word
    return float(max(35.0, min(160.0, 4.0 * elements_per_s)))


def envelope(audio, pitch, rate=DEFAULT_RATE, bandwidth=60.0,
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


def _reads_like_cw(text):
    """Does this decode look like a station, or like noise?

    Noise decoded at a threshold produces scattered single characters -- the
    ones with the fewest elements, E and T and I -- separated by spaces, and
    question marks where nothing matched. A station produces words. The
    difference is stark enough to test directly, and it is the last thing
    standing between a band scan and a list of frequencies where nothing is.
    """
    tokens = [t for t in text.split() if t]
    if len(tokens) < 2:
        return False
    unknown = text.count("?") / max(len(text), 1)
    if unknown > 0.25:
        return False
    singles = sum(1 for t in tokens if len(t) == 1)
    if singles / len(tokens) > 0.5:
        return False
    return sum(len(t) for t in tokens) / len(tokens) >= 1.8


def find_cw_signals(audio, rate=DEFAULT_RATE, lo=300.0, hi=2700.0,
                    min_db=16.0, min_wpm=8.0, max_wpm=55.0, min_chars=4):
    """Every CW signal in the passband, with how confident we are in each.

    With a wide filter one capture covers the whole passband, so a band scan can
    step in kilohertz rather than hertz. The hard part is not finding tones, it
    is not being fooled: an earlier version reported a signal at every frequency
    it looked at, all at the same strength and all sending at an implausible
    60 wpm.

    The test applied is the strongest one available -- try to decode it. A tone
    that yields characters at a human speed, with dahs about three times a dit,
    is a station. Anything else is a carrier, a birdie, or noise. Reusing the
    decoder also means the speed reported here is measured the same way as the
    speed reported everywhere else, rather than by a second, worse method.
    """
    import classic
    if len(audio) < rate:
        return []
    n = 8192
    spec = np.zeros(n // 2 + 1)
    frames = 0
    for i in range(0, len(audio) - n, n // 2):
        spec += np.abs(np.fft.rfft(audio[i:i + n] * np.hanning(n))) ** 2
        frames += 1
    if not frames:
        return []
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return []
    floor = float(np.median(spec[band])) or 1e-30

    peaks = []
    for i in np.where(band)[0][1:-1]:
        if spec[i] <= spec[i - 1] or spec[i] <= spec[i + 1]:
            continue
        db = 10 * np.log10(spec[i] / floor)
        if db >= min_db:
            peaks.append((float(freqs[i]), float(db)))
    peaks.sort(key=lambda t: -t[1])

    found, taken = [], []
    for hz, db in peaks:
        if any(abs(hz - t) < 80 for t in taken):
            continue                      # same signal, neighbouring bin
        env = envelope(audio, hz, rate, bandwidth=cw_bandwidth(20))
        snr = snr_estimate(env)
        norm, _, _ = normalise(env)
        text, info = classic.decode(norm)
        wpm, ratio = info.get("wpm"), info.get("ratio")
        letters = sum(c.isalnum() for c in text)
        if not wpm or not ratio:
            continue
        if not (min_wpm <= wpm <= max_wpm):
            continue
        if not (1.7 <= ratio <= 4.5):      # real CW: a dah is about three dits
            continue
        if letters < min_chars:
            continue
        if not _reads_like_cw(text):
            continue
        taken.append(hz)
        found.append({
            "audio_hz": round(hz, 1), "db": round(db, 1), "snr": round(snr, 1),
            "wpm": round(wpm, 1), "ratio": round(ratio, 2),
            "chars": letters, "sample": text[:36],
        })
    return sorted(found, key=lambda s: -s["snr"])
