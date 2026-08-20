"""The neural decoder: envelope frames in, characters out.

Deliberately small. This has to decode in real time on a cheap laptop in a park,
running off a battery, while the same machine also draws a waterfall and logs
QSOs -- so the budget is well under a megabyte of weights and a fraction of one
core, not whatever a GPU could manage.

The shape of the problem suggests the architecture. Timing is everything in CW
and the meaning of a gap depends on the gaps around it, so the network needs to
see a window of context: dilated convolutions widen that window cheaply. Reading
is then a sequence problem with no alignment between frames and characters,
which is what CTC exists for -- it lets us train on (audio, text) pairs without
ever labelling which frame belongs to which letter.
"""
import torch
import torch.nn as nn

# Everything the decoder can output. Index 0 is reserved for the CTC blank.
ALPHABET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./?=,-+"
BLANK = 0
CHARS = {c: i + 1 for i, c in enumerate(ALPHABET)}
INV = {i + 1: c for i, c in enumerate(ALPHABET)}
N_CLASSES = len(ALPHABET) + 1


def encode_text(text):
    return [CHARS[c] for c in text.upper() if c in CHARS]


def decode_greedy(logits):
    """Best path through the CTC output: take the top class per frame, collapse
    repeats, drop blanks."""
    best = logits.argmax(dim=-1).tolist()
    out, prev = [], None
    for k in best:
        if k != prev and k != BLANK:
            out.append(INV.get(k, ""))
        prev = k
    return "".join(out).strip()


class CWNet(nn.Module):
    def __init__(self, hidden=64, classes=N_CLASSES):
        super().__init__()
        # Dilated convolutions: each layer doubles how far back the network can
        # see, so seven of them span several seconds of audio -- long enough to
        # hold a whole character and the gaps on both sides of it -- while
        # costing far less than a wide kernel would.
        chans = [1, 32, 32, 48, 48, 64, 64, 64]
        layers = []
        for i in range(7):
            layers += [
                nn.Conv1d(chans[i], chans[i + 1], kernel_size=3,
                          padding=2 ** i, dilation=2 ** i),
                nn.BatchNorm1d(chans[i + 1]),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)
        # Bidirectional, because a gap is only interpretable once you have seen
        # what follows it. Live decoding therefore lags by a short window, which
        # is the price of reading a gap correctly rather than guessing at it.
        self.rnn = nn.GRU(chans[-1], hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Linear(hidden * 2, classes)

    def forward(self, x):
        """x: (batch, frames) envelope -> (batch, frames, classes) log-probs."""
        h = self.conv(x.unsqueeze(1)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.head(h).log_softmax(-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def hidden_of(state):
    """Read the GRU width out of a checkpoint.

    The width is a training choice and gets baked into the weights. Loading a
    checkpoint into the wrong shape fails with a shape error if you are lucky
    and silently mismatches if you are not, so it is read back rather than
    assumed.
    """
    for key in ("rnn.weight_ih_l0", "gru.weight_ih_l0"):
        w = state.get(key)
        if w is not None:
            return int(w.shape[0] // 3)      # GRU packs three gates per unit
    raise KeyError("no recurrent layer in this checkpoint; cannot tell its "
                   "width, and guessing loads the weights into the wrong shape")
