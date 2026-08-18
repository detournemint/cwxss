"""Regression tests.

Most of these are bugs that reached the air during a single evening of building
this against a real radio. The names say what broke, so a failure says what came
back.

    python3 tests/run.py
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cwxss"))

import classic, config, dsp, guess, lexicon, morse, rbn, score, stream, synth  # noqa: E402


class Timing(unittest.TestCase):
    """The definition everything else is measured against."""

    def test_paris_is_fifty_units(self):
        """The word that fixes words-per-minute. If this is wrong, every speed
        in the program is wrong and nothing else can be trusted."""
        units = sum(n for _, n in morse.to_units("PARIS")) + morse.GAP_WORD
        self.assertEqual(units, 50)

    def test_dit_length_at_known_speeds(self):
        self.assertAlmostEqual(morse.dit_seconds(20), 0.060, places=4)
        self.assertAlmostEqual(morse.dit_seconds(10), 0.120, places=4)

    def test_alphabet_round_trips(self):
        for ch, code in morse.CODE.items():
            self.assertEqual(morse.from_symbols(code), ch)

    def test_prosigns_are_single_characters(self):
        """<BT> is one character sent without internal gaps, not B then T."""
        seq = morse.to_units("<BT>")
        gaps = [n for on, n in seq if not on]
        self.assertTrue(all(g <= morse.GAP_ELEMENT for g in gaps))

    def test_word_gap_is_longer_than_character_gap(self):
        self.assertGreater(morse.GAP_WORD, morse.GAP_CHAR)
        self.assertGreater(morse.GAP_CHAR, morse.GAP_ELEMENT)


class Synthesis(unittest.TestCase):
    def test_duty_cycle_matches_real_cw(self):
        """Measured against an ARRL practice file: 0.42. If the synthesiser
        drifts from that, everything trained on it drifts too."""
        a = synth.render("CQ TEST DE K6XSS K", wpm=13, pitch=750, snr_db=30,
                         eff_wpm=13, seed=1)
        env = dsp.envelope(a, 750)
        e, _, _ = dsp.normalise(env)
        self.assertAlmostEqual(float((e > 0.5).mean()), 0.42, delta=0.10)

    def test_farnsworth_stretches_gaps_not_characters(self):
        """The defining property: the characters are sent at full speed and only
        the spaces between them grow. Stretching the characters as well would
        just be a slower speed, which is a different thing entirely.

        So the test is not "the clip got longer by some factor" -- that depends
        on how much of the text is key-down -- but "the key-down time is
        unchanged while the total time grew".
        """
        text = "CQ CQ DE K6XSS TEST"
        normal = synth.keyed_envelope(text, 20, synth.Fist.keyer())
        wide = synth.keyed_envelope(text, 20, synth.Fist.keyer(), eff_wpm=10)
        down_normal = float((normal > 0.5).sum())
        down_wide = float((wide > 0.5).sum())
        self.assertAlmostEqual(down_normal, down_wide, delta=down_normal * 0.02,
                               msg="characters must be sent at the same speed")
        self.assertGreater(len(wide), len(normal),
                           "the gaps must have grown")
        # and the extra time is all silence
        self.assertGreater((len(wide) - down_wide), (len(normal) - down_normal))

    def test_noise_lowers_measured_snr(self):
        quiet = dsp.snr_estimate(dsp.envelope(
            synth.render("TEST", snr_db=30, pitch=600, seed=1), 600))
        noisy = dsp.snr_estimate(dsp.envelope(
            synth.render("TEST", snr_db=0, pitch=600, seed=1), 600))
        self.assertGreater(quiet, noisy)

    def test_fists_differ_in_the_way_hands_do(self):
        rng = np.random.default_rng(0)
        spread = {}
        for name, f in (("keyer", synth.Fist.keyer(rng)),
                        ("good", synth.Fist.good_op(rng)),
                        ("rough", synth.Fist.rough_op(rng))):
            spread[name] = float(np.std([f.scale(True, 1) for _ in range(500)]))
        self.assertEqual(spread["keyer"], 0.0, "a machine keyer is exact")
        self.assertGreater(spread["rough"], spread["good"])


class Dsp(unittest.TestCase):
    def test_pitch_found_accurately_even_when_weak(self):
        for snr in (25, 10, 0, -6):
            a = synth.render("CQ DE K6XSS", wpm=20, pitch=655, snr_db=snr, seed=3)
            f, _ = dsp.find_pitch(a)
            self.assertIsNotNone(f, f"no pitch at {snr} dB")
            self.assertLess(abs(f - 655), 15, f"pitch off at {snr} dB")

    def test_pitch_is_none_when_there_is_not_enough_audio(self):
        """Returning a confident 0 Hz from 200 samples started a long wrong
        turn; not knowing has to be sayable."""
        f, sharp = dsp.find_pitch(np.zeros(200, dtype=np.float32))
        self.assertIsNone(f)

    def test_bandwidth_follows_speed(self):
        """A 20 wpm signal occupies about 70 Hz. Listening to 200 Hz of
        spectrum cost 10 dB on real off-air audio."""
        self.assertLess(dsp.cw_bandwidth(20), 100)
        self.assertGreater(dsp.cw_bandwidth(40), dsp.cw_bandwidth(15))
        self.assertGreaterEqual(dsp.cw_bandwidth(5), 35)

    def test_clean_signal_does_not_report_zero_snr(self):
        """A near-silent noise floor means a very clean signal. Reporting 0 dB
        for it -- as this did on an ARRL file -- says the opposite of the truth."""
        a = synth.render("TEST", wpm=20, pitch=600, snr_db=40, seed=1)
        self.assertGreater(dsp.snr_estimate(dsp.envelope(a, 600)), 15)


class ClassicDecoder(unittest.TestCase):
    def decode(self, text, **kw):
        kw.setdefault("pitch", 620)
        kw.setdefault("seed", 4)
        a = synth.render(text, **kw)
        e, _, _ = dsp.normalise(dsp.envelope(a, kw["pitch"]))
        return classic.decode(e)

    def test_clean_cw_is_read_exactly(self):
        for wpm in (13, 20, 30):
            got, _ = self.decode("CQ POTA DE K6XSS K", wpm=wpm, snr_db=25)
            self.assertGreater(score.accuracy("CQ POTA DE K6XSS K", got), 0.95,
                               f"{wpm} wpm")

    def test_speed_estimate_is_close(self):
        _, info = self.decode("CQ CQ DE K6XSS K6XSS K", wpm=22, snr_db=25)
        self.assertAlmostEqual(info["wpm"], 22, delta=4)

    def test_dit_estimate_survives_text_with_few_dahs(self):
        """Taking a median assumes an even mix of dits and dahs. EEE SSS HHH is
        almost all dits, and E and T alone break that assumption."""
        got, _ = self.decode("SEE HIS SITE", wpm=20, snr_db=25)
        self.assertGreater(score.accuracy("SEE HIS SITE", got), 0.7)

    def test_characters_carry_their_position_in_time(self):
        """Without this a sliding window cannot tell a newly heard character
        from one already read out, and prints BBOBB."""
        a = synth.render("CQ TEST", wpm=20, pitch=620, snr_db=25, seed=1)
        e, _, _ = dsp.normalise(dsp.envelope(a, 620))
        chars, _ = classic.decode_chars(e)
        self.assertTrue(chars)
        for ch, start, end in chars:
            self.assertLessEqual(start, end)
        starts = [s for _, s, _ in chars]
        self.assertEqual(starts, sorted(starts), "positions must increase")


class Streaming(unittest.TestCase):
    def stream(self, audio, chunk=1600, settle=True):
        sd = stream.StreamDecoder()
        for i in range(0, len(audio), chunk):
            sd.feed(audio[i:i + chunk])
        if settle:
            sd.feed(np.zeros(8000, dtype=np.float32))
        return sd

    def test_noise_produces_no_text(self):
        """A threshold detector always finds something. Pointed at an empty
        band it filled the screen with E and T and I."""
        rng = np.random.default_rng(0)
        sd = self.stream(rng.normal(0, 0.05, 8000 * 20).astype(np.float32),
                         settle=False)
        self.assertEqual(sd.committed.strip(), "")
        self.assertTrue(sd.quiet, "it should say why it is silent")

    def test_a_real_signal_still_gets_through(self):
        a = synth.render("CQ POTA DE K6XSS K", wpm=20, pitch=620, snr_db=16, seed=4)
        sd = self.stream(a)
        self.assertGreater(score.accuracy("CQ POTA DE K6XSS K", sd.committed), 0.85)

    def test_the_opening_of_a_transmission_is_not_clipped(self):
        """Squelching too eagerly skipped past the first second of every
        transmission: CQ POTA DE ... arrived as OTA DE ..."""
        a = synth.render("CQ POTA DE K6XSS K", wpm=20, pitch=620, snr_db=18, seed=4)
        self.assertTrue(self.stream(a).committed.strip().startswith("CQ"))

    def test_characters_are_not_emitted_twice(self):
        """A sliding window re-measures every character; boundaries that move by
        a frame read as new ones. A real QSO came out as BBOBB and SSUUNNYY."""
        a = synth.render("BOB SUNNY DAY", wpm=20, pitch=620, snr_db=22, seed=2)
        got = self.stream(a).committed
        self.assertNotIn("BBOB", got)
        self.assertNotIn("SSUNN", got)
        self.assertGreater(score.accuracy("BOB SUNNY DAY", got), 0.8)

    def test_filter_narrows_to_the_signal(self):
        a = synth.render("CQ DE K6XSS K", wpm=20, pitch=620, snr_db=20, seed=1)
        self.assertLess(self.stream(a).bandwidth, 110)


class Guessing(unittest.TestCase):
    def test_fills_a_single_missing_character(self):
        for token, want in (("R?GHT", "RIGHT"), ("SUN?Y", "SUNNY"),
                            ("ANT?NNA", "ANTENNA"), ("5N?", "5NN")):
            got, _ = guess.repair(token)
            self.assertEqual(got[0], want, token)

    def test_context_decides_between_equally_good_readings(self):
        """?RM is WARM before DAY and QRM before ON 40. The pattern fits both."""
        self.assertEqual(guess.repair("?RM DAY")[0][0], "WARM")
        self.assertEqual(guess.repair("?RM ON 40")[0][0], "QRM")

    def test_a_repair_is_never_presented_as_a_copy(self):
        _, marks = guess.repair("R?GHT")
        self.assertNotEqual(marks[0], "copied")

    def test_text_that_was_copied_is_left_alone(self):
        got, marks = guess.repair("CQ DE K6XSS K")
        self.assertEqual(" ".join(got), "CQ DE K6XSS K")
        self.assertTrue(all(m == "copied" for m in marks))

    def test_callsign_consensus_from_repetition(self):
        """Operators send their call two or three times. That is free
        redundancy and needs no dictionary."""
        got, _ = guess.repair("CQ DE VA7LXX VA7LXX VA?LXX K")
        self.assertEqual(got.count("VA7LXX"), 3)

    def test_a_wildcard_may_stand_for_a_digit(self):
        """Testing only a letter substitution made VA?LXX not look like a
        callsign at all, and consensus quietly did nothing."""
        got, _ = guess.repair("W1ABC W1ABC W?ABC")
        self.assertEqual(got.count("W1ABC"), 3)


class Macros(unittest.TestCase):
    def test_both_sets_exist_and_differ(self):
        cfg = config.load()
        self.assertIn("pota", cfg["sets"])
        self.assertIn("general", cfg["sets"])
        self.assertNotEqual(cfg["sets"]["pota"][0]["text"],
                            cfg["sets"]["general"][0]["text"])

    def test_placeholders_are_filled(self):
        cfg = dict(config.load(), call="K6XSS", state="CA", rst="5NN")
        self.assertEqual(config.expand("{his} {rst} {state} DE {call}", cfg, "W1ABC"),
                         "W1ABC 5NN CA DE K6XSS")

    def test_an_unset_field_leaves_no_token_behind(self):
        cfg = dict(config.load(), park="")
        self.assertEqual(config.expand("PARK IS {park}", cfg), "PARK IS")

    def test_an_older_flat_config_still_works(self):
        """Replacing an operator's own macros with the defaults on upgrade
        would be a poor way to repay them."""
        import json
        import tempfile
        old = {"call": "W1ABC",
               "messages": [{"key": "F1", "label": "x", "text": "CQ DE {call}"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(old, fh)
            path = fh.name
        prev = os.environ.get("CWXSS_CONFIG")
        os.environ["CWXSS_CONFIG"] = path
        try:
            import importlib
            importlib.reload(config)
            cfg = config.load()
            self.assertEqual(cfg["sets"]["pota"][0]["text"], "CQ DE {call}")
        finally:
            if prev is None:
                os.environ.pop("CWXSS_CONFIG", None)
            else:
                os.environ["CWXSS_CONFIG"] = prev
            import importlib
            importlib.reload(config)
            Path(path).unlink(missing_ok=True)


class Rbn(unittest.TestCase):
    def test_parses_a_real_spot_line(self):
        line = ("DX de W1NT-6-#: 14025.10  K6XSS          CW    12 dB  "
                "19 WPM  CQ      0415Z")
        m = rbn.SPOT.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m["dx"], "K6XSS")
        self.assertEqual(int(m["snr"]), 12)
        self.assertEqual(int(m["wpm"]), 19)

    def test_ignores_lines_that_are_not_spots(self):
        for line in ("Hello, K6XSS! Connected.", "Local users: 494", ""):
            self.assertIsNone(rbn.SPOT.search(line))

    def test_only_our_own_callsign_counts(self):
        mon = rbn.RbnMonitor("K6XSS", log=lambda *_: None)
        other = rbn.SPOT.search(
            "DX de PI4CC-#:   7010.40  9A2V           CW    31 dB  22 WPM  CQ  0413Z")
        self.assertNotEqual(other["dx"].upper(), mon.call)


class Vocabulary(unittest.TestCase):
    def test_the_common_abbreviations_are_known(self):
        for w in ("CQ", "DE", "TNX", "QTH", "QRM", "RST", "73", "ES", "HR", "UR"):
            self.assertIn(w, lexicon.WORDS, w)

    def test_frequency_ranking_prefers_common_words(self):
        self.assertLess(lexicon.frequency_rank("CQ"), lexicon.frequency_rank("RETIRED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
