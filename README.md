# cwxss

A CW decoder and keyboard keyer you run from a browser. Built for a cheap laptop
in a park: no GPU, modest CPU, and a web interface so the operating position can
be a phone propped against the rig.

## What it does

- **Decodes CW** from the radio's audio, live, with speed and pitch tracking
- **Sends CW from the keyboard**, by keying the rig's own keying line
- **POTA and general macro sets**, F1-F10, switchable
- **Best guess** — fills in what the decoder could not read, and marks every
  repair so a guess is never mistaken for a copy
- **Who hears me** — live Reverse Beacon Network spots of your own callsign
- **Saves real signals** with their metadata, for improving the decoder

## Two decoders

A classic timing decoder and a small neural one, side by side, because the
second has to earn its place. The classic decoder is exact on clean machine-sent
CW and falls apart on a human fist; the neural one is trained for the cases it
fails. Run `python3 cwxss/eval.py --model models/cw.pt` to compare them on
identical clips.

## Running it

    python3 cwxss/server.py --demo                 # synthetic signal, no radio
    python3 cwxss/server.py --device plughw:2,0 \
        --rig 127.0.0.1:4532 --keyline /dev/ttyUSB1

## Keying

Keying is done by toggling a serial control line -- the electrical equivalent of
a straight key -- which is what flrig, fldigi and N1MM do.

Hamlib's `send_morse` is tried as a fallback but does not work on every rig. On
a Yaesu it becomes the CAT `KY` command, and on an FT-991A `KY` only replays
stored keyer memories: it cannot accept arbitrary text. Hamlib asks the rig
whether the keyer buffer is free, the rig answers `?;` because it does not
implement that query, and hamlib concludes the radio is busy forever.

For an FT-991A: **menu 060 PC KEYING = DTR**, and use the second serial port the
radio presents (the first is CAT).

## Training data

There is no public corpus of real, labelled, off-air CW -- every project in this
space generates its own. `synth.py` does that here, and models the things that
actually break decoders: hand-sent timing, Farnsworth spacing, semi-automatic
keys, fading, a competing station, and static crashes.

Verified against reality: a 13 wpm ARRL practice file and our synthetic
equivalent have the same duty cycle to within 2%.
