"""Continuous decoding: audio arriving forever, text coming out as it goes.

Decoding a finished recording is a different problem from decoding a live band.
Live, the tone moves when the operator drifts or you tune, the speed changes
with every station, and there is no end of file to wait for. This holds a rolling
window and re-reads it as audio arrives.

Characters are emitted only once they are settled. A character at the very edge
of the window may still gain another element, so committing it immediately means
printing "N" and then having to take it back when the next dit arrives and makes
it "K". Text is committed a short way behind the live edge, which is exactly how
a human operator reads: slightly behind, and right.
"""
import time

import numpy as np

import classic
import dsp
import guess
import neural

WINDOW_S = 12.0            # how much history to keep and re-read
# Silence long enough to count as the end of an over. Shorter than the gap
# between overs in a QSO, longer than the gap between words at 8 wpm.
BLOCK_BREAK_S = 20.0
HISTORY_BLOCKS = 600       # blocks kept in memory
HISTORY_VIEW = 40          # blocks sent with each update
COMMIT_LAG_S = 1.2         # how far behind the edge text is treated as settled

# A threshold detector always finds something. Point it at an empty band and it
# reads the noise floor as a stream of dits, which is why an unsquelched decoder
# fills the screen with E and T and I. None of that is information, and it
# buries the moments that are. Before committing anything, the signal has to
# look like Morse: loud enough to stand out, sent at a speed a human could send,
# and with dahs about three times a dit.
MIN_SNR_DB = 9.0
MIN_WPM, MAX_WPM = 8.0, 55.0
MIN_RATIO, MAX_RATIO = 1.8, 4.5
MIN_ELEMENTS = 6
GRACE_S = 4.0      # audio kept in hand while deciding whether it is a signal
# Above this much gap stretch the sender is using Farnsworth spacing, which is
# where the classic decoder inserts a word break between every letter and where
# the model is far ahead. Anywhere in 4 to 6 gives the same result on the nine
# recordings this was measured on.
STRETCH_PREFER_NEURAL = 4.5


class StreamDecoder:
    def __init__(self, rate=dsp.DEFAULT_RATE, frame_rate=dsp.FRAME_RATE,
                 window_s=WINDOW_S, model="models/cw.onnx"):
        self.rate = rate
        self.frame_rate = frame_rate
        self.window = int(window_s * frame_rate)
        self.env = np.zeros(0, dtype=np.float32)
        # The window is kept as audio, not as envelope frames. The filter width
        # follows the sending speed, and mixing frames measured at 80 Hz with
        # frames measured at 60 Hz makes a window that is not a measurement of
        # anything. Recomputing from audio costs one pass over 12 seconds of
        # samples, which is nothing, and keeps the window consistent.
        self.audio = np.zeros(0, dtype=np.float32)
        self.audio_tail = np.zeros(0, dtype=np.float32)   # partial frame
        # Pitch detection needs a second or two of audio; chunks arriving live
        # are a fraction of that, so keep a rolling history to search in.
        self.audio_hist = np.zeros(0, dtype=np.float32)
        self.pitch_hist_s = 3.0
        self.pitch = None
        self.committed = ""
        # A timestamped transcript, kept as blocks rather than one string.
        #
        # From a net control operator explaining why he runs a decoder at all:
        # he copies by ear and uses the screen to look *back* when a word gets
        # away from him. A rolling 4000-character string cannot answer "what did
        # he send five minutes ago", which for someone working a roster is the
        # entire point. Blocks break on silence, so the transcript reads as
        # separate overs rather than one unbroken wall.
        self.history = []
        self.pending = ""
        self.info = {}
        # Absolute position of the oldest frame still in the window, and how far
        # we have already read out. Both count frames since the stream began, so
        # they stay meaningful as the window slides.
        self.frames_dropped = 0
        self.read_to = 0
        self.quiet = "waiting for a signal"
        self.bandwidth = 80.0
        # The trained decoder runs beside the classic one rather than replacing
        # it. It is better where the classic decoder fails -- a hand fist,
        # Farnsworth spacing, a fading signal -- and no better where it does
        # not, so showing both lets the operator see which to believe.
        self.net = neural.NeuralDecoder(model)
        self.neural_text = ""
        self.neural_conf = 0.0
        # Which decoder to believe, and why. Measured on nine ARRL recordings:
        # the model wins where the gaps are stretched and the classic decoder
        # wins where the timing is exact, and choosing between them by that one
        # measurement scores 74.3% where the better single decoder scores 70.9.
        self.stretch = 0.0
        self.prefer = "classic"

    def feed(self, audio):
        """Add audio. Returns text newly committed by this chunk."""
        if audio is None or len(audio) == 0:
            return ""
        chunk = np.asarray(audio, dtype=np.float32)
        buf = np.concatenate([self.audio_tail, chunk])
        self.audio_hist = np.concatenate([self.audio_hist, chunk])[
            -int(self.pitch_hist_s * self.rate):]

        # Re-find the tone regularly: operators drift, and the operator tunes.
        found, sharp = dsp.find_pitch(self.audio_hist, self.rate)
        if found and sharp > 6.0:
            self.pitch = found if self.pitch is None else (
                0.7 * self.pitch + 0.3 * found)          # follow slowly

        if self.pitch is None:
            self.audio_tail = buf[-int(0.05 * self.rate):]
            return ""

        step = int(self.rate / self.frame_rate)
        usable = (len(buf) // step) * step
        if usable == 0:
            self.audio_tail = buf
            return ""
        self.audio_tail = buf[usable:]
        keep = int(self.window * self.rate / self.frame_rate)
        before = self.audio.size
        self.audio = np.concatenate([self.audio, buf[:usable]])[-keep:]
        dropped_samples = max(0, before + usable - self.audio.size)
        self.frames_dropped += dropped_samples // step

        # Match the filter to the speed being sent. A wide filter admits the
        # neighbouring station as well as the noise; on a busy band there may be
        # half a dozen signals inside the radio's own filter, and this one was
        # three times wider than a 20 wpm signal occupies.
        self.bandwidth = dsp.cw_bandwidth(self.info.get("wpm"))
        self.env = dsp.envelope(self.audio, self.pitch, self.rate,
                                bandwidth=self.bandwidth,
                                frame_rate=self.frame_rate)
        return self._reread()

    def _reread(self):
        """Re-read the window and commit whatever has settled since last time.

        The whole window is decoded each pass, because a character near the edge
        can still change. Each decoded character carries the frames it occupies,
        so committing is a matter of position rather than of guessing which part
        of the text is new -- string matching cannot distinguish a newly heard
        character from one already read out, and gets it wrong every time the
        same letter appears twice.
        """
        if self.env.size < self.frame_rate:
            return ""
        norm, _, _ = dsp.normalise(self.env)
        chars, info = classic.decode_chars(norm, offset=self.frames_dropped)
        self.info = info
        self.quiet = self._why_quiet(info, chars)
        if self.quiet:
            self.pending = ""
            # Keep read_to moving so a backlog of noise is not dumped on screen
            # the moment a signal appears -- but leave a few seconds of grace.
            # Advancing it right up to the edge threw away the opening of every
            # transmission, because the first second of a signal is exactly when
            # there is not yet enough of it to pass the test: "CQ POTA DE ..."
            # arrived as "OTA DE ...".
            grace = int(GRACE_S * self.frame_rate)
            self.read_to = max(self.read_to,
                               self.frames_dropped + norm.size - grace)
            return ""

        edge = self.frames_dropped + norm.size - int(COMMIT_LAG_S * self.frame_rate)
        # Commit on where a character STARTS, not where it ends. Re-reading a
        # sliding window re-measures every character, and an end boundary that
        # moves by a frame or two between passes reads as a new character --
        # which is why a real QSO came out as "BBOBB" and "SSUUNNYY".
        # A start position is stable: the same character keeps the same one.
        # Even a start position wanders slightly: each pass re-derives the
        # threshold and the dit length from a marginally different window, so
        # run boundaries move by a frame or two. Require a new character to
        # begin at least half a dit beyond the last one committed, which is far
        # more than that wander and far less than a real gap.
        margin = max(2.0, (info.get("dit") or 6) * 0.5)
        fresh = [(c, st, end) for c, st, end in chars
                 if end <= edge and st > self.read_to + margin]
        added = "".join(c for c, _, _ in fresh)
        if fresh:
            self.read_to = max(st for _, st, _ in fresh)
            self.committed = (self.committed + added)[-4000:]
            self._record(added)
        self.pending = "".join(c for c, _, end in chars if end > edge).strip()
        return added

    def _record(self, added):
        """File newly committed text under a timestamp."""
        now = time.time()
        if self.history and now - self.history[-1]["last"] < BLOCK_BREAK_S:
            self.history[-1]["text"] += added
            self.history[-1]["last"] = now
        else:
            self.history.append({"at": now, "last": now, "text": added})
            del self.history[:-HISTORY_BLOCKS]

    def transcript(self):
        """The whole session, timestamped, as plain text."""
        out = []
        for b in self.history:
            stamp = time.strftime("%H:%M:%S", time.gmtime(b["at"]))
            out.append(f"{stamp}Z  {b['text'].strip()}")
        return "\n".join(out)

    def _why_quiet(self, info, chars):
        """Empty when the signal is worth decoding, otherwise the reason not to."""
        snr = dsp.snr_estimate(self.env)
        if snr < MIN_SNR_DB:
            return f"no signal ({snr:.0f} dB)"
        wpm = info.get("wpm")
        if not wpm or not (MIN_WPM <= wpm <= MAX_WPM):
            return f"speed implausible ({wpm} wpm)"
        ratio = info.get("ratio")
        if not ratio or not (MIN_RATIO <= ratio <= MAX_RATIO):
            return f"not Morse-like (dah:dit {ratio:.1f})" if ratio else "no timing"
        if sum(1 for c, _, _ in chars if c.strip()) < MIN_ELEMENTS:
            return "too little to read"
        return ""

    def state(self):
        norm, _, _ = dsp.normalise(self.env) if self.env.size else (self.env, 0, 0)
        # The repaired text is sent alongside the raw copy, never instead of it.
        # An operator has to be able to see what was actually heard.
        # The model reads the whole window at once: it has no notion of
        # committing, and re-reading is cheap enough not to need one.
        if self.net.available and not self.quiet and self.env.size > 100:
            norm, _, _ = dsp.normalise(self.env)
            self.neural_text, self.neural_conf = self.net.decode_scored(norm)
            level = classic.threshold(norm)
            seq = [r for r in classic.runs(norm, level) if r[1] >= 2]
            self.stretch = classic.gap_stretch(seq, self.info.get("dit"))
            self.prefer = ("neural" if self.stretch >= STRETCH_PREFER_NEURAL
                           else "classic")
        toks, marks = guess.repair(self.committed[-600:])
        return {
            "pitch": round(self.pitch, 1) if self.pitch else None,
            "wpm": self.info.get("wpm"),
            "snr": round(dsp.snr_estimate(self.env), 1) if self.env.size else 0.0,
            "envelope": [round(float(v), 3) for v in norm[-600:]],
            "text": self.committed[-2000:],
            # Recent blocks only; the full session is at /transcript, so a long
            # net does not push a megabyte through the socket every update.
            "history": [{"at": int(b["at"]), "text": b["text"]}
                        for b in self.history[-HISTORY_VIEW:]],
            "pending": self.pending,
            "guessed": [{"w": w, "m": m} for w, m in zip(toks, marks)],
            "quiet": self.quiet,
            "bandwidth": round(self.bandwidth),
            "neural": self.neural_text,
            "neural_conf": round(self.neural_conf, 2),
            "neural_ok": self.net.available,
            "neural_error": self.net.error,
            "stretch": round(self.stretch, 1),
            "prefer": self.prefer,
            "best": (self.neural_text if self.prefer == "neural" and self.neural_text
                     else self.committed[-2000:]),
        }
