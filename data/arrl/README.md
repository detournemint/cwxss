# ARRL W1AW code practice files

Paired audio and text, which makes them the only readily available *labelled*
real CW recordings.

    http://www.arrl.org/{SPEED}-wpm-code-archive
    /files/file/Morse/Archive/{SPEED} WPM/{YYMMDD}_{SPEED}WPM.mp3
    /files/file/Morse/Archive/{SPEED} WPM/{YYMMDD}_{SPEED}.txt

Speeds: 5, 7.5, 10, 13, 15, 18, 20, 25, 30, 35, 40 wpm. Tone 750 Hz. Updated
fortnightly. At 15 wpm and below the character speed is 15 wpm Farnsworth.

## What they are good for

Verifying the decoder against text we did not write: pitch tracking, speed
estimation, Farnsworth handling, and the character table. Our measured duty
cycle on a 13 wpm file matches our synthesiser exactly (0.40 both), which says
the timing model is right.

## What they are not

Representative of the air. They are studio-generated and machine-sent:

    ARRL 13 wpm file    envelope SNR 60 dB (capped)   tone peak 71 dB
    our "clean" synth   envelope SNR 26 dB            tone peak 47 dB

They are cleaner than anything a POTA operator will ever hear, and being
machine-sent they contain none of the hand-sent timing that decoders actually
fail on. Training on them would teach the model that CW is easy.

## Licence

ARRL states: "Reproduction of material from any ARRL web page without written
permission is strictly prohibited." There is no open licence. Fine to download
and evaluate against privately; not something to train a distributed model on
without asking them.

The way around this is also the better engineering: use these files to *measure*
and tune the synthesiser, then train on synthetic audio. No ARRL content ends up
in the model.

## Better labelled real audio

- **Our own transmissions.** We know exactly what was sent, and it goes through
  a real transmitter and receiver.
- **W1AW off the air**: 1.8025, 3.5815, 7.0475, 14.0475, 18.0775, 21.0675,
  28.0675, 50.350, 147.555 MHz. Slow code 5-15 wpm, fast code 10-35 wpm,
  bulletins 18 wpm; the text is from QST and cited at the start of each session.
  Real propagation, real noise, and still labelled.
