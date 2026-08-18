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
      SNR rather than on the shape of the distribution itself. The neural
      decoder already handles Farnsworth at 100%, which is the better answer
      for now.

- [ ] **Self-supervised pretraining on unlabelled real CW.** The highest-value
      idea outstanding. There is no public corpus of labelled off-air CW and
      there never will be, but unlabelled recordings are free -- `harvest.py`
      collects them. A wav2vec-style pretraining stage would let those
      recordings improve the model without a single transcript, which is the
      whole obstacle to using live audio. *(Needs a corpus first: needs radio.)*

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

- [ ] **Run the overnight harvester.** `harvest.py` sweeps, finds, records and
      runs both decoders over each capture, unattended. Built and deployed,
      never run. *(Needs radio.)*

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

## Notes

Things tried and rejected, so they are not tried again:

- **Rejecting short key-downs as noise.** A 20 ms blip cannot be a 60 ms dit,
  but the dit estimate is derived partly from noise, so the threshold lands
  above some real dits and eats them. Turned WARM DAY back into ?RM DAY.
- **Per-character confidence gating.** Dropping individual low-scoring
  characters cost 9 points during a deep fade: confidence falls across the whole
  passage and the characters discarded there are often right. Gating the passage
  costs nothing and works.
