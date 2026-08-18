"""Morse code: the alphabet, and the timing that turns it into a signal.

Timing is defined in "dit units". At any speed a dit is one unit, a dah is
three, the gap inside a character is one, between characters three, and between
words seven. Everything downstream -- generating audio, decoding it, judging a
decoder -- is measured in those units, so they live here once.

The standard word PARIS is exactly 50 units, which is what fixes words-per-minute
to a dit length: 20 WPM means 1000 units a minute, so a dit is 60 ms.
"""

# The table is written as dit/dah strings because that is how it is read aloud
# and how it is checked. Everything else is derived from it.
CODE = {
    "A": ".-",     "B": "-...",   "C": "-.-.",   "D": "-..",    "E": ".",
    "F": "..-.",   "G": "--.",    "H": "....",   "I": "..",     "J": ".---",
    "K": "-.-",    "L": ".-..",   "M": "--",     "N": "-.",     "O": "---",
    "P": ".--.",   "Q": "--.-",   "R": ".-.",    "S": "...",    "T": "-",
    "U": "..-",    "V": "...-",   "W": ".--",    "X": "-..-",   "Y": "-.--",
    "Z": "--..",
    "0": "-----",  "1": ".----",  "2": "..---",  "3": "...--",  "4": "....-",
    "5": ".....",  "6": "-....",  "7": "--...",  "8": "---..",  "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.",  "=": "-...-",
    "+": ".-.-.",  "-": "-....-", ":": "---...", "'": ".----.", '"': ".-..-.",
    "(": "-.--.",  ")": "-.--.-", "@": ".--.-.", "!": "-.-.--", "&": ".-...",
    ";": "-.-.-.", "$": "...-..-",
}
DECODE = {v: k for k, v in CODE.items()}

# Prosigns are sent as one character with no internal gap. Written between angle
# brackets so they survive a round trip through text without being mistaken for
# the letters they are made of.
PROSIGNS = {
    "<AR>": ".-.-.",    # end of message
    "<SK>": "...-.-",   # end of contact
    "<BT>": "-...-",    # break / new paragraph
    "<KN>": "-.--.",    # go ahead, named station only
    "<AS>": ".-...",    # wait
    "<BK>": "-...-.-",  # break in
}
for _p, _c in PROSIGNS.items():
    DECODE.setdefault(_c, _p)

DIT, DAH = 1, 3
GAP_ELEMENT, GAP_CHAR, GAP_WORD = 1, 3, 7
PARIS_UNITS = 50            # dit units in "PARIS ", which defines WPM


def dit_seconds(wpm):
    """Length of one dit at a given speed."""
    if wpm <= 0:
        raise ValueError("wpm must be positive")
    return 60.0 / (PARIS_UNITS * wpm)


def encode(text):
    """Text to dit/dah strings, one per character. Unknown characters are
    dropped rather than guessed at -- inventing a symbol would corrupt training
    labels silently."""
    out, i, up = [], 0, text.upper()
    while i < len(up):
        if up[i] == "<":                       # a prosign, maybe
            end = up.find(">", i)
            token = up[i:end + 1] if end > 0 else ""
            if token in PROSIGNS:
                out.append((token, PROSIGNS[token]))
                i = end + 1
                continue
        ch = up[i]
        if ch == " ":
            out.append((" ", ""))
        elif ch in CODE:
            out.append((ch, CODE[ch]))
        i += 1
    return out


def to_units(text):
    """Text to a key-down/key-up timeline in dit units.

    Returns a list of (on, units) with no trailing gap: the caller decides how
    much silence to leave at the end.
    """
    seq = []

    def gap(units):
        if seq and not seq[-1][0]:
            seq[-1] = (False, seq[-1][1] + units)   # merge adjacent silence
        else:
            seq.append((False, units))

    first = True
    for ch, code in encode(text):
        if ch == " ":
            gap(GAP_WORD - GAP_CHAR if not first else GAP_WORD)
            continue
        if not first:
            gap(GAP_CHAR)
        first = False
        for j, sym in enumerate(code):
            if j:
                gap(GAP_ELEMENT)
            seq.append((True, DIT if sym == "." else DAH))
    while seq and not seq[0][0]:
        seq.pop(0)
    while seq and not seq[-1][0]:
        seq.pop()
    return seq


def from_symbols(symbols):
    """A dit/dah string to its character, or '' if it is not one we know."""
    return DECODE.get(symbols, "")
