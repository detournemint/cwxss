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

import classic, config, dsp, guess, lexicon, morse, neural, qsolog  # noqa: E402
import rbn, score                                                          # noqa: E402
import selftrain                                                            # noqa: E402
import stream, synth                                                       # noqa: E402


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


class DecoderChoice(unittest.TestCase):
    """Which decoder to believe, measured rather than assumed."""

    def stretch_of(self, **kw):
        kw.setdefault("snr_db", 22)
        a = synth.render("CQ POTA DE K6XSS K TEST TEST", pitch=620, seed=5, **kw)
        e, _, _ = dsp.normalise(dsp.envelope(a, 620, bandwidth=dsp.cw_bandwidth(20)))
        level = classic.threshold(e)
        seq = [r for r in classic.runs(e, level) if r[1] >= 2]
        dit, _ = classic.estimate_dit([n for on, n in seq if on])
        return classic.gap_stretch(seq, dit)

    def test_standard_timing_reads_low(self):
        """Character gaps at 3 dits and word gaps at 7 put this near 2-4."""
        self.assertLess(self.stretch_of(wpm=20), 5.0)
        self.assertGreater(self.stretch_of(wpm=20), 1.0)

    def test_farnsworth_reads_high(self):
        """Stretching the gaps is what this measures, so it must see it."""
        self.assertGreater(self.stretch_of(wpm=20, eff_wpm=7),
                           self.stretch_of(wpm=20))

    def test_the_threshold_sits_between_them(self):
        """Measured on nine ARRL recordings: standard timing lands at 2.0-3.7
        and Farnsworth at 6.1-16.6, so anywhere in 4 to 6 separates them."""
        self.assertGreater(stream.STRETCH_PREFER_NEURAL, 4.0)
        self.assertLess(stream.STRETCH_PREFER_NEURAL, 6.0)

    def test_extreme_farnsworth_crosses_the_threshold(self):
        """The ARRL 5 wpm file is 15 wpm characters with 3:1 spacing and
        measures 16.6. That is the case the whole mechanism exists for: the
        classic decoder scores 35% on it and the model 53%, so the choice has
        to actually fire rather than merely trend upwards."""
        self.assertGreater(self.stretch_of(wpm=15, eff_wpm=5),
                           stream.STRETCH_PREFER_NEURAL)

    def test_choosing_beats_committing_to_either(self):
        """Selection is worth more than either decoder alone -- on the nine
        ARRL recordings, +5.7 points over always-classic and +3.4 over
        always-model. If that ever stops being true the mechanism is dead
        weight, so the ordering is asserted rather than left to the eye."""
        fast = self.stretch_of(wpm=28)              # classic's ground
        slow = self.stretch_of(wpm=15, eff_wpm=5)   # the model's ground
        self.assertLess(fast, stream.STRETCH_PREFER_NEURAL)
        self.assertGreater(slow, stream.STRETCH_PREFER_NEURAL)

    def test_no_gaps_is_not_a_stretch(self):
        self.assertEqual(classic.gap_stretch([], 10), 0.0)
        self.assertEqual(classic.gap_stretch([(True, 5)], 0), 0.0)


class NeuralAccumulation(unittest.TestCase):
    """The model has to remember what it read, not just the last window."""

    def stream(self, text="CQ POTA DE K6XSS K TEST DE W1AW QRZ", wpm=18):
        a = synth.render(text, wpm=wpm, pitch=640, snr_db=20, seed=6)
        d = stream.StreamDecoder(model="models/cw.onnx")
        n = int(0.5 * dsp.DEFAULT_RATE)
        for i in range(0, a.size - n, n):
            d.feed(a[i:i + n])
            d.state()
        return d

    def test_timed_decode_agrees_with_plain_decode(self):
        """The positions are extra information, not a different reading."""
        net = neural.NeuralDecoder("models/cw.onnx")
        if not net.available:
            self.skipTest(net.error)
        a = synth.render("CQ DE K6XSS K", wpm=20, pitch=640, snr_db=20, seed=3)
        env, _, _ = dsp.normalise(
            dsp.envelope(a, 640, bandwidth=dsp.cw_bandwidth(20)))
        timed, _ = net.decode_timed(env)
        self.assertEqual("".join(c for c, _ in timed), net.decode(env))

    def test_positions_land_inside_the_window(self):
        net = neural.NeuralDecoder("models/cw.onnx")
        if not net.available:
            self.skipTest(net.error)
        a = synth.render("CQ DE K6XSS K", wpm=20, pitch=640, snr_db=20, seed=3)
        env, _, _ = dsp.normalise(
            dsp.envelope(a, 640, bandwidth=dsp.cw_bandwidth(20)))
        timed, _ = net.decode_timed(env)
        self.assertTrue(timed)
        for _, pos in timed:
            self.assertGreaterEqual(pos, 0)
            self.assertLess(pos, len(env))

    def test_the_model_keeps_more_than_the_last_window(self):
        """Measured over seventeen ARRL recordings, showing only the current
        window scored 39.5% against the transcript and accumulating scored
        82.2%. The model was not reading badly; it was forgetting."""
        d = self.stream()
        net = neural.NeuralDecoder("models/cw.onnx")
        if not net.available:
            self.skipTest(net.error)
        self.assertGreaterEqual(len(d.neural_committed), len(d.neural_text))

    def test_committing_does_not_repeat_the_window(self):
        """The model re-reads the whole window every time. Appending its
        output would say everything several times over."""
        d = self.stream()
        if not d.neural_committed:
            self.skipTest("model produced nothing")
        self.assertLess(d.neural_committed.count("K6XSS"), 4)


class SignalFinding(unittest.TestCase):
    """Which decoder gets a vote on whether a signal exists."""

    class FakeNet:
        available = True

        def __init__(self, text=""):
            self.text = text

        def decode(self, env, *a, **kw):
            return self.text

    def clip(self):
        return synth.render("CQ POTA DE K6XSS K TEST DE W1AW QRZ", wpm=18,
                            pitch=640, snr_db=14, seed=4)

    def test_noise_is_rejected_even_with_a_model_voting(self):
        """Giving a second decoder a vote must not reintroduce the forty
        phantom signals a band sweep once reported."""
        rng = np.random.default_rng(3)
        for rms in (0.03, 0.10, 0.20):
            a = (rng.standard_normal(8000 * 20) * rms).astype(np.float32)
            self.assertEqual(
                dsp.find_cw_signals(a, 8000, net=self.FakeNet("")), [])

    def test_a_model_that_reads_it_is_enough(self):
        """A station on 14049.5, spotted by two RBN receivers forty miles
        away, arrived 47 dB over the floor and was discarded because the
        classic decoder made noise of it while the model read a CQ and two
        callsigns. Either decoder reading language is enough."""
        found = dsp.find_cw_signals(
            self.clip(), dsp.DEFAULT_RATE,
            net=self.FakeNet("CQ DE K6XSS K TEST DE W1AW"))
        self.assertTrue(found)

    def test_it_still_works_with_no_model_at_all(self):
        """The model is optional; a station running without it must still get
        a band scan."""
        self.assertTrue(dsp.find_cw_signals(self.clip(), dsp.DEFAULT_RATE))

    def test_a_babbling_model_cannot_invent_a_signal(self):
        """The timing gates run before either language test, so a model
        emitting plausible text over noise still cannot conjure a station."""
        rng = np.random.default_rng(9)
        a = (rng.standard_normal(8000 * 20) * 0.05).astype(np.float32)
        self.assertEqual(
            dsp.find_cw_signals(a, 8000,
                                net=self.FakeNet("CQ DE W1AW K TEST QRZ ES")),
            [])


class TwoStageScan(unittest.TestCase):
    """A scan finds candidates; confirming them is a separate, slower job."""

    def test_noise_produces_no_candidates_even_without_the_language_test(self):
        """The timing gates, not the language test, are what keep noise out of
        the fast pass -- noise does not key at a human speed with a three to
        one ratio. If that stopped being true the fast pass would flood."""
        rng = np.random.default_rng(5)
        for rms in (0.03, 0.10, 0.20):
            a = (rng.standard_normal(8000 * 20) * rms).astype(np.float32)
            self.assertEqual(
                dsp.find_cw_signals(a, 8000, require_language=False), [])

    def test_the_language_test_needs_more_than_a_scan_step(self):
        """Measured on a real off-air station: found at 20 seconds, invisible
        at 15 or fewer. A scan pausing four seconds per step cannot apply this
        test, which is why the scan does not try to."""
        a = synth.render("CQ POTA DE K6XSS K", wpm=22, pitch=640, snr_db=16,
                         seed=2)
        short = a[:int(3 * dsp.DEFAULT_RATE)]
        self.assertEqual(dsp.find_cw_signals(short, dsp.DEFAULT_RATE), [])


class SelfTraining(unittest.TestCase):
    """Pseudo-labels are only worth having if they are right."""

    class FakeNet:
        """Stands in for the model so these tests need no ONNX file."""
        available = True

        def __init__(self, text):
            self.text = text

        def decode(self, env, *a, **kw):
            return self.text

    def test_disagreement_is_rejected(self):
        """The whole filter rests on this. Two decoders that fail differently
        agreeing is evidence; either one alone is not."""
        env = np.zeros(400, dtype=np.float32)
        text, why = selftrain.consider(
            env, self.FakeNet("CQ POTA DE K6XSS K"))
        self.assertIsNone(text)

    def test_short_decodes_are_rejected(self):
        """Two decoders agreeing on four characters is easy by accident."""
        self.assertLess(len("CQ K"), selftrain.MIN_CHARS)

    def test_agreement_threshold_is_strict(self):
        """Measured on synthetic clips with known truth: unfiltered labels are
        67% accurate and agreement-filtered ones 98.5%. Loosening this is what
        turns self-training into the model teaching itself its own mistakes."""
        self.assertGreaterEqual(selftrain.MIN_AGREEMENT, 0.80)

    def test_noise_that_parses_is_still_rejected(self):
        """A band sweep once reported forty signals where there were none, all
        of them noise that decoded into plausible-looking tokens. The same
        content test guards the training set."""
        self.assertFalse(dsp._reads_like_cw("EEI?E E? EESEI ITSEH TT E"))


class ErrorProsign(unittest.TestCase):
    """An operator cancelling a word and sending it again.

    From a net control operator: "CW ops will misspell a word, send a bunch of
    Es, try again... BUT by the time they get the word spelled correctly I've
    forgotten the first part of the sentence."
    """

    def marks(self, text):
        tok, mk = guess.repair(text)
        return dict(zip(tok, mk)), tok, mk

    def test_the_cancelled_word_is_struck(self):
        _, tok, mk = self.marks("WX HR IS WEATNER EEEEEEEE WEATHER FB")
        self.assertEqual(mk[tok.index("WEATNER")], "struck")
        self.assertEqual(mk[tok.index("WEATHER")], "copied")

    def test_dits_split_into_separate_tokens_still_count(self):
        """Whether a run of dits arrives as one token or eight depends on how
        the sender spaced them, which is not something to rely on."""
        _, tok, mk = self.marks("RIG IS FT991 E E E E E E FT991A")
        self.assertEqual(mk[tok.index("FT991")], "struck")

    def test_hh_is_the_written_prosign(self):
        _, tok, mk = self.marks("NAME IS BOB HH ROB")
        self.assertEqual(mk[tok.index("BOB")], "struck")

    def test_real_words_made_of_dits_are_left_alone(self):
        """SEE, HE, IS and SHE are all dits. Striking the word before them
        would be worse than not having the feature at all."""
        _, tok, mk = self.marks("I SEE HE IS HIS SHE ES IT")
        self.assertNotIn("struck", mk)

    def test_a_signal_report_is_not_a_correction(self):
        _, tok, mk = self.marks("UR RST 559 559 ES NAME")
        self.assertNotIn("struck", mk)

    def test_a_short_run_is_not_a_correction(self):
        """Three dits is the letter S, or a decoder having a bad moment."""
        _, tok, mk = self.marks("TNX FER EEE CALL")
        self.assertNotIn("struck", mk)

    def test_two_corrections_walk_back_two_words(self):
        """They try, fail, try again, fail again. Each cancel has to strike a
        different attempt or the transcript blames one word twice."""
        _, tok, mk = self.marks("QTH IS PORTLNAD EEEEE PORTLND EEEEE PORTLAND")
        self.assertEqual(mk[tok.index("PORTLNAD")], "struck")
        self.assertEqual(mk[tok.index("PORTLND")], "struck")
        self.assertEqual(mk[tok.index("PORTLAND")], "copied")


class Sixty(unittest.TestCase):
    def test_sixty_metres_is_channels_not_a_segment(self):
        """Sweeping 5332-5405 in 2.5 kHz steps would spend almost all of it on
        spectrum where no amateur may transmit."""
        import harvest
        self.assertIn("60", harvest.CHANNELS)
        self.assertNotIn("60", harvest.SEGMENTS)
        self.assertEqual(len(harvest.CHANNELS["60"]), 5)


class FistSpread(unittest.TestCase):
    """A machine and a hand send differently, and it can be measured."""

    def spread_of(self, fist, wpm=20, snr=20, seed=5):
        a = synth.render("CQ POTA DE K6XSS K TEST DE W1AW QRZ ES TU", wpm=wpm,
                         pitch=640, snr_db=snr, fist=fist, seed=seed)
        e, _, _ = dsp.normalise(
            dsp.envelope(a, 640, bandwidth=dsp.cw_bandwidth(20)))
        level = classic.threshold(e)
        seq = [r for r in classic.runs(e, level) if r[1] >= 2]
        dit, dah = classic.estimate_dit([n for on, n in seq if on])
        return classic.fist_spread(seq, dit, dah)

    def test_a_keyer_is_regular(self):
        rng = np.random.default_rng(1)
        self.assertLess(self.spread_of(synth.Fist.keyer(rng)),
                        stream.SPREAD_PREFER_NEURAL)

    def test_a_rough_fist_is_not(self):
        rng = np.random.default_rng(1)
        self.assertGreater(self.spread_of(synth.Fist.rough_op(rng)),
                           stream.SPREAD_PREFER_NEURAL)

    def test_the_threshold_clears_every_benchmark_recording(self):
        """Every ARRL practice file measures between 0.014 and 0.076 because
        they are machine-sent. The threshold sits above all of them, so adding
        this signal cannot change a single benchmark result -- which is what
        made it safe to add on evidence the benchmark cannot provide."""
        self.assertGreater(stream.SPREAD_PREFER_NEURAL, 0.08)
        self.assertLess(stream.SPREAD_PREFER_NEURAL, 0.15)

    def test_too_few_elements_is_not_a_verdict(self):
        self.assertEqual(classic.fist_spread([], 6, 18), 0.0)
        self.assertEqual(classic.fist_spread([(True, 6)], 6, 18), 0.0)
        self.assertEqual(classic.fist_spread([(True, 6)] * 20, None, None), 0.0)


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


class AutoLog(unittest.TestCase):
    """Reading a contact out of decoded CW.

    Ten contacts make a park activation and each one has to be entered while
    the next station is already calling, so this reads the exchange off the
    transcript instead.
    """

    def read(self, text, mine="K6XSS"):
        return qsolog.read_exchange(text, my_call=mine)

    def test_a_temperature_is_not_a_callsign(self):
        """Found by running auto-log over a real recording: a ragchew about the
        weather -- "WARM DAY ES SUNNY 21C" -- offered 21C as the station
        worked. Every callsign prefix contains a letter; 21C has none."""
        self.assertFalse(qsolog.valid_call("21C"))
        self.assertEqual(
            self.read("HR BOB WARM DAY ES SUNNY 21C ES 73")["call"], "")

    def test_unusual_but_real_prefixes_still_pass(self):
        """Tightening the rule must not throw out the callsigns that made the
        loose version tempting: a digit-first prefix is perfectly ordinary."""
        for c in ("4X4AB", "9A2V", "W1AW/7", "VE7ABC", "KH6XYZ"):
            self.assertTrue(qsolog.valid_call(c), c)

    def test_a_clean_exchange(self):
        r = self.read("K6XSS DE N0ABC N0ABC 579 MN TU 73")
        self.assertEqual(r["call"], "N0ABC")
        self.assertEqual(r["rst_rcvd"], "579")
        self.assertEqual(r["state"], "MN")

    def test_our_own_call_is_not_the_contact(self):
        """We send our own callsign more than anyone else does."""
        r = self.read("CQ POTA DE K6XSS K6XSS K6XSS K DE W1ABC W1ABC")
        self.assertEqual(r["call"], "W1ABC")

    def test_nothing_worked_is_refused(self):
        """Calling CQ into an empty band must not produce a log entry."""
        self.assertEqual(self.read("CQ CQ CQ DE K6XSS K6XSS K")["call"], "")

    def test_de_is_not_delaware(self):
        """Half the state abbreviations are ordinary CW. DE means 'from' and
        would otherwise land in the state field of every contact ever logged;
        HI is laughter, IN OR ME OH are words, AR and SK are prosigns."""
        self.assertEqual(self.read("K6XSS DE W1ABC BK")["state"], "")

    def test_the_state_is_taken_from_the_exchange_position(self):
        """Their state follows their report, which is what distinguishes it
        from the same two letters appearing anywhere else."""
        r = self.read("W1ABC 599 CA R 599 TX TU")
        self.assertEqual(r["state"], "TX")

    def test_park_to_park(self):
        r = self.read("K6XSS DE KK6IK 599 CA K-1234 TU")
        self.assertEqual(r["their_park"], "K-1234")

    def test_5nn_is_599(self):
        """Nobody sends 599 at speed; they send 5NN."""
        self.assertEqual(self.read("K6XSS DE W1ABC 5NN TU")["rst_rcvd"], "599")

    def test_hearing_a_call_once_is_flagged(self):
        """Logged anyway -- a missed contact cannot be recovered and Undo is
        one click -- but the operator is told to check it."""
        self.assertLess(self.read("K6XSS DE W1ABC BK")["confidence"], 1.0)
        self.assertEqual(
            self.read("K6XSS DE W1ABC W1ABC 599 TU")["confidence"], 1.0)


class Transcript(unittest.TestCase):
    """Looking back at what was sent five minutes ago."""

    def test_blocks_break_on_silence(self):
        d = stream.StreamDecoder(model="models/cw.onnx")
        d._record("CQ DE K6XSS")
        d.history[-1]["last"] -= stream.BLOCK_BREAK_S + 1
        d._record("K6XSS DE W1ABC")
        self.assertEqual(len(d.history), 2)

    def test_continuous_sending_stays_one_block(self):
        d = stream.StreamDecoder(model="models/cw.onnx")
        d._record("CQ ")
        d._record("DE K6XSS")
        self.assertEqual(len(d.history), 1)
        self.assertEqual(d.history[0]["text"], "CQ DE K6XSS")

    def test_the_transcript_is_timestamped(self):
        d = stream.StreamDecoder(model="models/cw.onnx")
        d._record("CQ DE K6XSS")
        self.assertRegex(d.transcript(), r"^\d\d:\d\d:\d\dZ  CQ DE K6XSS")


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


class QsoLog(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.path = tempfile.mktemp(suffix=".jsonl")
        self.log = qsolog.Log(self.path)

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def test_a_contact_is_written_immediately(self):
        """An activation ends when the battery does. A log held in memory is a
        log you can lose."""
        self.log.add("W1ABC", freq_hz=7030000)
        self.assertTrue(Path(self.path).exists())
        self.assertEqual(len(qsolog.Log(self.path).qsos), 1)

    def test_nonsense_is_refused(self):
        for bad in ("NOTACALL", "", "12345", "K"):
            q, err = self.log.add(bad)
            self.assertIsNone(q, bad)
            self.assertTrue(err)

    def test_real_callsigns_are_accepted(self):
        for good in ("W1ABC", "K6XSS", "VE7XYZ", "G0ABC", "JA1XY", "W1ABC/7"):
            self.assertTrue(qsolog.valid_call(good), good)

    def test_a_dupe_is_the_same_station_same_band_same_day(self):
        """Hunters chase an activator across bands all afternoon. Calling that
        a dupe would refuse contacts that count."""
        self.log.add("W1ABC", freq_hz=7030000)
        self.assertEqual(len(self.log.worked("W1ABC", band="40m")), 1)
        self.assertEqual(len(self.log.worked("W1ABC", band="20m")), 0)
        self.assertEqual(len(self.log.worked("K5DXX", band="40m")), 0)

    def test_activation_needs_ten(self):
        for i in range(9):
            self.log.add(f"W1AB{chr(65+i)}", freq_hz=7030000)
        self.assertFalse(self.log.summary()["activated"])
        self.assertEqual(self.log.summary()["needed"], 1)
        self.log.add("K5DXX", freq_hz=7030000)
        self.assertTrue(self.log.summary()["activated"])

    def test_undo_removes_from_the_file_too(self):
        self.log.add("W1ABC", freq_hz=7030000)
        self.log.add("K5DXX", freq_hz=7030000)
        self.log.remove_last()
        self.assertEqual(len(qsolog.Log(self.path).qsos), 1)

    def test_adif_is_well_formed_and_carries_the_pota_fields(self):
        self.log.add("W1ABC", freq_hz=7030000, state="MA",
                     my_park="US-1178", their_park="US-4567")
        out = qsolog.adif(self.log.today(), "K6XSS", "US-1178")
        self.assertIn("<EOH>", out)
        self.assertEqual(out.count("<EOR>"), 1)
        for field in ("<CALL:5>W1ABC", "<BAND:3>40m", "<MODE:2>CW",
                      "<MY_SIG:4>POTA", "<MY_SIG_INFO:7>US-1178",
                      "<SIG_INFO:7>US-4567", "<STATION_CALLSIGN:5>K6XSS"):
            self.assertIn(field, out, field)

    def test_band_is_derived_from_frequency(self):
        for hz, band in ((7030000, "40m"), (14030000, "20m"),
                         (3550000, "80m"), (999, "")):
            self.assertEqual(qsolog.band_of(hz), band)


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
