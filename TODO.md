# TODO

Ordered by what would help an operator most, with what is already known about
each. Anything marked *needs radio* cannot be worked on without a rig connected.

## Decoding

- [ ] **Extreme Farnsworth.** The classic decoder scores 35% on ARRL's 5 wpm
      practice file: 15 wpm characters with 3:1 spacing. Every letter decodes
      correctly and a word break is inserted between each one, because gaps are
      classified as fixed multiples of a dit (element < 2, word >= 5).

      Measured on that file the gaps are 1, 16 and 38 dits, so every character
      gap lands past the word threshold.

      **Two attempts have been made and both were reverted.** Learning the
      boundaries from the largest steps in the sorted gap distribution fixes
      Farnsworth (63.7% -> 100%) and costs 9.5 points overall: a noisy gap
      distribution has spurious steps, and QSB fell 33 points and static
      crashes 41. Requiring stronger evidence before overriding made it worse
      still (72.5%), and three-way clustering was worse again -- on a rough
      fist it produced thresholds of 0.6 and 1.9 dits and decoded nothing.

      The lesson is that the gap distribution is only trustworthy when the
      signal is clean, so any future attempt should gate on measured envelope
      SNR rather than on the shape of the distribution itself.

      **A fourth attempt, in the training distribution, was also reverted.**
      Sending ratio is not the gap stretch the decoder measures, and
      calibrating them explains the failure exactly: 1.5:1 reads as 4.2, 3.2:1
      as 8.9, 6:1 as 16.9. The ARRL 5 wpm file measures 16.6, so it is a 6:1
      send, and training capped at 3.2:1 -- the model had genuinely never seen
      a gap that long. Widening to 7:1 and retraining made it worse, and worse
      on the target rather than merely elsewhere:

      | | live | widened |
      |---|---|---|
      | 5 wpm file | 53.3% | **50.0%** |
      | 10 wpm file | 62.7% | **55.7%** |
      | synthetic | 97.7% | 96.4% |
      | real, chosen | 74.3% | 73.1% |

      179,757 parameters over 9000 steps is a fixed budget, and spreading it
      wider buys coverage by spending depth: the tail at 6:1 was too sparse to
      learn from while still diluting the middle. **A next attempt needs more
      capacity or more steps before more spread**, and should oversample the
      extreme rather than reach it with a thin tail.

- [ ] **Self-supervised pretraining on unlabelled real CW.** The highest-value
      idea outstanding. There is no public corpus of labelled off-air CW and
      there never will be, but unlabelled recordings are free -- `harvest.py`
      collects them.

      **The filter is built and measured** (`selftrain.py`). Naive
      pseudo-labelling teaches a model to repeat itself, so the test is not
      confidence but *agreement between the classic decoder and the model*,
      which fail in close to uncorrelated ways. Against known truth on 120
      synthetic clips:

      | pseudo-labels | accuracy |
      |---|---|
      | unfiltered | 67.0% |
      | agreement-filtered | **98.5%** |

      25% yield, 27 of 30 survivors above 90%. The fine-tuning loop is
      deliberately unwritten until there is real audio to point it at.
      *(Corpus started: two captures, see Harvesting.)*

- [ ] **Spectrogram input instead of an envelope.** DeepCW feeds the network 65
      frequency bins across 400-1200 Hz and lets it find the tone itself; we
      pre-select a pitch and collapse to one number per frame, so our accuracy
      is capped by the pitch detector. Head to head our envelope model matched
      theirs, so this is speculative rather than obvious -- but it would remove
      a dependency on getting the pitch right.

- [ ] **CBAM attention**, per the CCBC paper. The published Morse networks
      converge on CNN + BiGRU + CTC with an attention module; ours is that
      without the attention.

- [ ] **Accumulate neural text across windows.** The model reads the current
      12-second window and shows only that, while the classic decoder
      accumulates. Comparing them over a long transmission is misleading as a
      result.

## Station

- [ ] **Find stations button.** `dsp.find_cw_signals` is built and validated --
      it finds two signals 400 Hz apart, reports nothing on noise, and uses the
      decoder itself as the test of whether a tone is a station. It is not
      wired to a button, and a band sweep moves the VFO. *(Needs radio.)*

- [x] **Run the harvester.** Done, and it exposed three bugs that made every
      sweep incapable of finding anything.

      1. **Only the classic decoder voted** on whether a tone was a station. A
         signal 47 dB over the floor passed every timing test and was discarded
         because classic read `E SII T? E E B? ILF` where the model read
         `CO DBK RR DK G5AQV KL5NQ` -- a CQ and two callsigns. A harvester
         built to collect training audio *for the model* was throwing away
         exactly the signals only the model could read.
      2. **Four seconds cannot judge language.** The language test needs three
         or four tokens; a scan step collects one or two. The same recording:
         0 signals at 4s and 15s, 1 signal at 20s. Scanning and confirming are
         now separate jobs.
      3. **Blind sweeping is a lottery** regardless. One sweep passed 14025 two
         minutes before a station started calling CQ there and reported the
         band empty.

- [ ] **Harvesting: tune the RBN targeting.** `--rbn` asks nearby skimmers
      where to point the radio and produced two recordings in nine minutes
      against zero from three hours of sweeping. Hit rate was 2 of 7, and the
      misses were all spotted at 5-12 dB while the keeper was at 29 dB -- a
      skimmer's SNR predicts what this antenna can copy. Raising the floor is
      not quite right, though: the second keeper came from a 5 dB spot of a
      station that was actually strong and one skimmer simply heard badly.
      Counting *how many* skimmers heard it is probably the better signal.
      Corpus lives on redwood at `~/cwdata`.

- [ ] **Xiegu X6100 on real hardware.** Profile written (hamlib 3087, 19200
      baud, DTR keying on the single CAT port, `Can send Morse: N` so hamlib
      cannot key it). Untested against the radio. *(Needs radio.)*

- [ ] **Auto-answer.** The decoder already knows when someone sends our
      callsign. Prompting -- or offering a one-key reply -- would close the loop
      between decoding and the log.

- [ ] **Callsign lookup.** QRZ or a local database, to fill in state and name
      rather than typing them at 25 wpm.

## Interface

- [ ] **Band and frequency controls.** There is no way to change band from the
      browser; it is done over CAT or at the rig.

- [ ] **A diagnostics bundle**, as ft8xss has: versions, devices, live state and
      recent errors, redacted, for a bug report.

- [ ] **Tests for the server itself.** The WebSocket actions have no coverage;
      the decoders and the log are well covered and the plumbing around them
      is not.

## From an operator who runs a decoder on his own net

Net control for a Sunday 60m net, on why he keeps a decoder running. Worth
recording because it describes a use we had not designed for: he copies by ear
at 70-80% and uses the screen as a **backstop**, not a replacement.

> Most often, CW ops will misspell a word, send a bunch of "Es", try again, and
> then yet again and finally get it right - BUT by the time they get the word
> spelled correctly I've forgotten the first part of the sentence.

Done: cancelled words are struck rather than printed as a wall of Es; 60m is
five channels rather than a segment; net vocabulary (NCS, QNI, QND, traffic).

Still open, in his order of pain:

- [ ] **Scrollback that survives the QSO.** Text is capped at the last 2000
      characters with no way to reach back five minutes. For net control
      tracking a roster this is the whole feature.
- [ ] **Timestamps in the transcript**, so a look-back has an anchor.
- [ ] **Recording from the browser.** He offered to record his net; that should
      be one button, not an ssh session.

He judges decoders against **CWTY (WD6CNF)** and **CwGet (UA9OV)**, not fldigi.
CWTY has no working download link any more, which is part of why this is worth
finishing. His net is also the corpus that would unblock self-training:
scheduled, regular, and with a control operator calling a roster, so the
callsigns are known in advance rather than guessed.

## Notes

Things tried and rejected, so they are not tried again:

- **Rejecting short key-downs as noise.** A 20 ms blip cannot be a 60 ms dit,
  but the dit estimate is derived partly from noise, so the threshold lands
  above some real dits and eats them. Turned WARM DAY back into ?RM DAY.
- **Anchored text edits that match nothing.** Five entries meant for this file
  were written against a heading that did not exist and silently did nothing,
  while the script reported success. The same bug had already cost a CSS rule
  earlier. Any scripted edit must assert that the text changed.
- **Per-character confidence gating.** Dropping individual low-scoring
  characters cost 9 points during a deep fade: confidence falls across the whole
  passage and the characters discarded there are often right. Gating the passage
  costs nothing and works.
