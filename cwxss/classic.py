"""The classic CW decoder: threshold, measure, classify.

This is the twenty-line-detector idea done properly, and it is the baseline the
neural decoder has to beat. It works by measuring how long the key is down and
how long it is up, working out how long a dit is, and reading off the rest in
multiples of that.

Its weakness is structural rather than a matter of tuning: every decision is a
threshold on one duration in isolation. Given a run of elements that a human
would read instantly from context, it has no context to appeal to.
"""
import numpy as np

from morse import from_symbols


def threshold(env):
    """Split an envelope into key-down and key-up.

    Otsu's method: pick the level that best separates the values into two
    groups. A fixed threshold fails the moment the signal fades, and the
    midpoint between min and max is set by whichever noise spike was loudest.
    """
    if env.size == 0:
        return 0.5
    hist, edges = np.histogram(env, bins=64)
    centres = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.5
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


def runs(env, level):
    """Envelope to alternating (on, frames) runs."""
    keyed = env > level
    if keyed.size == 0:
        return []
    edges = np.flatnonzero(np.diff(keyed.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(keyed)]])
    return [(bool(keyed[bounds[i]]), int(bounds[i + 1] - bounds[i]))
            for i in range(len(bounds) - 1)]


def gap_stretch(seq, dit):
    """How far the gaps are stretched beyond a dit.

    Standard timing puts character gaps at 3 dits and word gaps at 7, so this
    sits near 2-4. Farnsworth stretches the gaps while leaving the characters
    alone, and it climbs: ARRL's 5 wpm practice measures 16.

    It is worth measuring because it says which decoder to believe. The fixed
    thresholds in this file fail exactly when the gaps are stretched, and that
    is exactly where the trained model is strongest.
    """
    if not dit:
        return 0.0
    gaps = np.array([n for on, n in seq if not on], dtype=float)
    wide = gaps[gaps > dit * 1.8]
    return float(np.median(wide) / dit) if wide.size else 0.0


def estimate_dit(on_runs):
    """Length of a dit, in frames, from the key-down runs.

    Two-means on the durations. Taking the shortest run instead would hand the
    estimate to a single noise glitch, and taking the median assumes an even mix
    of dits and dahs, which no real text has -- E and T alone break it.
    """
    d = np.asarray([n for n in on_runs if n > 0], dtype=float)
    if d.size == 0:
        return None, None
    if d.size < 4 or d.max() / max(d.min(), 1) < 1.6:
        return float(np.median(d)), float(np.median(d) * 3)   # all one kind
    lo, hi = np.percentile(d, 20), np.percentile(d, 80)
    for _ in range(25):
        near_lo = np.abs(d - lo) <= np.abs(d - hi)
        if not near_lo.any() or near_lo.all():
            break
        lo_new, hi_new = d[near_lo].mean(), d[~near_lo].mean()
        if abs(lo_new - lo) < 1e-6 and abs(hi_new - hi) < 1e-6:
            break
        lo, hi = lo_new, hi_new
    return float(lo), float(hi)


def decode_runs(seq, dit, offset=0):
    """Runs to characters, given a dit length in frames.

    Returns [(char, start_frame, end_frame)]. The frame positions are what make
    live decoding possible: without them a sliding window cannot tell a newly
    decoded character from one it has already read out.
    """
    out, symbols, start, pos = [], "", None, offset
    for on, n in seq:
        if on:
            if start is None:
                start = pos
            symbols += "." if n < dit * 2 else "-"
            pos += n
            continue
        if n < dit * 2:                  # gap inside a character
            pos += n
            continue
        if symbols:
            out.append((from_symbols(symbols) or "?", start, pos))
            symbols, start = "", None
        if n >= dit * 5:                 # gap between words
            out.append((" ", pos, pos + n))
        pos += n
    if symbols:
        out.append((from_symbols(symbols) or "?", start, pos))
    return out


def text_of(chars):
    return "".join(c for c, _, _ in chars).strip()


# A key-down shorter than this fraction of a dit is not an element. Noise
# crossing the threshold for two or three frames was being read as E after E
# between real words, which is most of what makes a decoder look useless.
MIN_ELEMENT = 0.45


def drop_blips(seq, dit):
    """Remove key-downs too short to be elements, and close the gaps.

    A blip has to be merged into the surrounding silence rather than deleted,
    or the gaps either side of it read as two short gaps instead of one long
    one -- turning a space between words into a space between characters.
    """
    out = []
    for on, n in seq:
        if on and n < dit * MIN_ELEMENT:
            on = False                      # it was silence with a tick in it
        if out and out[-1][0] == on:
            out[-1] = (on, out[-1][1] + n)
        else:
            out.append((on, n))
    return out


def decode_chars(env, min_run=2, offset=0):
    """Envelope to timed characters. Returns ([(char, start, end)], info)."""
    level = threshold(env)
    seq = [r for r in runs(env, level) if r[1] >= min_run]
    if not seq:
        return [], {"level": level, "dit": None, "wpm": None}
    dit, dah = estimate_dit([n for on, n in seq if on])
    if not dit:
        return [], {"level": level, "dit": None, "wpm": None}
    # Rejecting short key-downs as noise was tried here and made things worse:
    # the dit estimate is itself derived partly from noise, so the threshold
    # lands above some real dits and eats them. On live off-air audio it turned
    # WARM DAY back into ?RM DAY and cost 6 points on clean synthetic. The
    # measurement has to be trusted less, not filtered harder.
    from dsp import FRAME_RATE
    wpm = 60.0 / (50 * dit / FRAME_RATE)
    return decode_runs(seq, dit, offset), {
        "level": float(level), "dit": dit, "dah": dah,
        "ratio": (dah / dit) if dit else None, "wpm": round(wpm, 1),
    }


def decode(env, min_run=2):
    """Envelope to text. Returns (text, info)."""
    level = threshold(env)
    seq = [r for r in runs(env, level) if r[1] >= min_run]
    if not seq:
        return "", {"level": level, "dit": None, "wpm": None, "elements": 0}
    dit, dah = estimate_dit([n for on, n in seq if on])
    if not dit:
        return "", {"level": level, "dit": None, "wpm": None, "elements": 0}
    from dsp import FRAME_RATE
    wpm = 60.0 / (50 * dit / FRAME_RATE) if dit else None
    return text_of(decode_runs(seq, dit)), {
        "level": float(level), "dit": dit, "dah": dah,
        "ratio": (dah / dit) if dit else None,
        "wpm": round(wpm, 1) if wpm else None,
        "elements": sum(1 for on, _ in seq if on),
    }
