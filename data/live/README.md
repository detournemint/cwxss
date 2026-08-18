# Off-air recordings

Real CW captured by `harvest.py` from an FT-991A in Benicia, California
(CM88VC). Each `.wav` is 8 kHz mono; each `.json` carries what both decoders
made of it, the measured speed and SNR, and -- for later captures -- the
callsign the Reverse Beacon Network attributed to that frequency.

That last field is the useful one. There is no public corpus of labelled
off-air CW, and an RBN spot names the station, so these arrive with a callsign
already attached rather than guessed at.

| capture | found by | notes |
|---|---|---|
| `cw-20260818-031858` | blind sweep | a QSO: "HR BOB ... WARM DAY ES SUNNY 21C" |
| `cw-20260818-042217` | blind sweep | "KE0M DE KE4EW" |
| `cw-20260818-213952` | RBN, W1AW/7 on 14050 | both decoders read `KK6IK 5NN` |
| `cw-20260818-214552` | RBN, W1AW/7 on 14050 | legible: `CQ ... W1AW/7 ... CQ` |

These are not transcribed, so they cannot be used for supervised training as
they stand. `selftrain.py` exists to make them usable: it decodes each clip with
both the classic decoder and the model and keeps only clips where the two agree,
which measured 98.5% label accuracy against known truth versus 67% unfiltered.
Decoder agreement on the first RBN capture was 0.37, well below the 0.85 that
filter requires, so the corpus is not yet large or clean enough to train on.

Recording on amateur bands is unrestricted; these are public transmissions.
