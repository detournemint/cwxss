# cwxss

A CW decoder and keyboard keyer you run from a browser. Built for a cheap laptop
in a park: no GPU, no X server, and a web interface so the operating position can
be a phone propped against the rig.

    ./setup.sh --service
    # open http://<the machine>:8074

## Why another CW decoder

Existing decoders are good at machine-sent CW and poor at everything else. That
is not a complaint about any particular program, it is what the community says
about all of them, and it is measurable. Our own classic timing decoder, which
is a fair implementation of how they all work:

| condition | accuracy |
|---|---|
| clean keyer, 25 dB | 100% |
| **good operator, hand sent, 20 dB** | **99%** |
| **rough fist, 20 dB** | **72%** |
| Farnsworth 20/12 | 92% |
| weak, 3 dB | 45% |
| static crashes | 49% |

Look at the third row. That is a *strong, clean* signal — 20 dB — that decodes at
72% for no reason other than a human sent it. Most CW on the air is hand sent.

The problem is structural rather than a matter of tuning: a threshold decoder
judges every element in isolation, against a duration. A human reads a gap in the
context of the gaps around it, and knows that `K6XSS` is a callsign. So this
project does three things about it.

**A filter matched to the signal.** A 20 wpm CW signal occupies about 70 Hz. Most
software listens to far more, and everything extra is noise and the station next
door. On live off-air audio, narrowing from 200 Hz to 70 Hz was worth **10 dB** —
the difference between `'E QIEL E 7RL E EDE ?E4EW-'` and `'DE KE4EW = GA JOE BK'`.

**A best guess, clearly marked.** `R?GHT` is `RIGHT`; nothing else fits. `?RM` is
`WARM` before `DAY` and `QRM` before `ON 40`. A callsign sent three times can be
reconciled against itself. Every repair is shown differently from what was
actually copied, because a decoder that silently prints its guesses has told you
something you cannot check.

**A neural decoder, on trial.** A 180 KB model reading the envelope, running at
113× real time on one CPU core. It runs *beside* the classic decoder rather than
replacing it, and `eval.py` scores both on identical clips. It is worth having
where the classic decoder fails — weak signals, static — and is not worth having
where the classic decoder is already exact. Both are shown; neither is trusted by
default.

## What it does

- Decodes CW live, tracking pitch and speed, with a squelch so an empty band
  produces silence rather than a screen of `E`
- **Sends CW from the keyboard** by keying the rig's keying line
- POTA and general macro sets, F1–F10, switchable
- **Who hears me** — live Reverse Beacon Network spots of your own callsign
- Saves real signals with their metadata, for improving the decoder
- Verifies its own transmissions: reports watts out, and says so when the key
  line moved but the radio never keyed

## Setup

    git clone https://github.com/detournemint/cwxss
    cd cwxss
    ./setup.sh --service      # or ./setup.sh --check to diagnose

The script installs dependencies, lists your audio devices and marks which looks
like a radio rather than a webcam, works out which serial port is CAT and which
is the keying line, tests CAT, offers to set the radio's PC KEYING menu item, and
writes the config and a systemd service.

Try it with no radio at all:

    python3 cwxss/server.py --demo

### Keying

Keying toggles a serial control line — the electrical equivalent of a straight
key — which is what flrig, fldigi and N1MM do.

It deliberately does **not** rely on hamlib's `send_morse`. On a Yaesu that
becomes the CAT `KY` command, and on an FT-991A `KY` only replays stored keyer
memories: it cannot accept text. Hamlib asks the rig whether the keyer buffer is
free, the rig answers `?;` because it does not implement that query, and hamlib
concludes the radio is busy — permanently. It is offered as a fallback for rigs
where it works.

Two things are needed and neither is obvious:

- the **keying port**, which on a radio presenting two serial ports is usually
  the second one; the first is CAT
- the radio set to accept it. On an FT-991A: **menu 060 PC KEYING = DTR**. Off
  from the factory, and with it off the software sends perfectly while no RF
  comes out at all

## Credits, prior art, and what was taken

Nothing here is unprecedented, and where an idea came from someone else it is
named. No code or model weights from any of these projects are included.

### Other CW decoders

| project | licence | what it is |
|---|---|---|
| [f4exb/morseangel](https://github.com/f4exb/morseangel) | MIT | PyQt + PyTorch desktop decoder |
| [e04/web-deep-cw-decoder](https://github.com/e04/web-deep-cw-decoder) · [engine](https://github.com/e04/deepcw-engine) | AGPL-3.0 | browser decoder, spectrogram input |
| [ag1le/deepmorse-decoder](https://github.com/ag1le/deepmorse-decoder) · [LSTM_morse](https://github.com/ag1le/LSTM_morse) | GPL-3.0 | CNN-LSTM-CTC, 27.8 h training set |
| [netom/MorseNet](https://github.com/netom/MorseNet) | — | RNN + CTC research |
| [souryadey/morse-dataset](https://github.com/souryadey/morse-dataset) | — | synthetic Morse datasets for ML |
| [flrig / fldigi](http://www.w1hkj.com/flrig-help/cw_keyer.html) | GPL | where the control-line keying approach comes from |

**What was taken from DeepCW: the idea of not guessing.** Below about -6 dB it
falls silent while ours produced `CQ GDTL DDXR RRI 5W` -- equally fluent whether
reading a signal or noise. CTC reports a probability for every frame and we were
discarding it. No code or weights were used: DeepCW is AGPL-3.0 and this is
GPL-3.0, and adopting its model would extend network-use source disclosure to
everyone running this.

**What was taken from flrig and fldigi: how to key a radio.** They toggle a
serial control line rather than relying on hamlib's `send_morse`, and having
tried the alternative the reason is plain -- see the Keying section.

### Research

- **CTC** — Graves et al., *Connectionist Temporal Classification: Labelling
  Unsegmented Sequence Data with Recurrent Neural Networks*, ICML 2006. What
  makes it possible to train on (audio, text) pairs without ever labelling which
  frame belongs to which character.
- **Prefix beam search** — Graves & Jaitly 2014; Hannun's write-up of the
  algorithm. Implemented in `beam.py`, with the CW vocabulary as the language
  model. Measured here at about a point where the vocabulary matches and nothing
  where it does not; the honest numbers are in that file's history.
- **CCBC / DeepMorse** — the published Morse-recognition networks converge on
  CNN + BiGRU + CTC, with attention added
  ([DeepMorse](https://www.researchgate.net/publication/333793763_DeepMorse_A_Deep_Convolutional_Learning_Method_for_Blind_Morse_Signal_Detection_in_Wideband_Wireless_Spectrum),
  [detection and recognition in wideband spectrum](https://www.researchgate.net/publication/363794531_Detection_and_recognition_based_on_machine_learning_and_deep_learning_for_Morse_signal_in_wide-band_wireless_spectrum)).
  This project uses the same family without the attention module.
- **Morse Code Datasets for Machine Learning** — Dey, Chugg & Beerel,
  [arXiv:1807.04239](https://arxiv.org/pdf/1807.04239). Establishes what this
  project also found: there is no public corpus of real labelled CW, so everyone
  synthesises.
- **Otsu's method** (1979) — used to split the envelope into key-down and
  key-up in `classic.py`, in place of a fixed threshold.

### Data and services

- **[ARRL W1AW code practice files](http://www.arrl.org/code-practice-files)** —
  the only readily available CW recordings with exact transcripts, used for
  evaluation only. ARRL's material is not openly licensed: nothing fetched is
  redistributed or trained on.
- **[Reverse Beacon Network](https://reversebeacon.net/)** — skimmer spots of
  your own callsign.
- **[hamlib](https://hamlib.github.io/)** — CAT control.

### How this compares

Run head to head on identical clips, our 710 KB model and DeepCW's 15 MB one
were level: identical on clean, weak, Farnsworth and QRN cases, DeepCW slightly
ahead on a fading signal, ours ahead on a semi-automatic key.

What is different here is not the decoder, which is the least novel part of this
project. Those projects all decode only. This one keys a transmitter, and is a
station -- rig control, macros, skimmer spots, band scanning -- that runs on a
laptop in a park with no internet.

## Training data

There is no public corpus of real, labelled, off-air CW. Every project in this
space generates its own, and so does this one — `synth.py` models what actually
defeats decoders rather than what is easy to simulate: hand-sent timing,
Farnsworth spacing, semi-automatic keys, fading, a competing station 150 Hz away,
and static crashes.

Checked against reality: a 13 wpm ARRL practice file and our synthetic equivalent
have the same duty cycle to within 2%.

## Tests

    python3 tests/run.py

37 cases, most of them bugs that reached the air.

## Licence

GPL-3.0.
