"""Fill in what the decoder could not read, and be honest about which is which.

A decoder that prints R?GHT when it means RIGHT is asking the operator to do
work a computer can do better. But a decoder that silently prints RIGHT when it
actually heard R?GHT has told a lie, and the operator has no way to know which
words to trust.

So every repair is marked. The text carries its own confidence, and the
interface shows guesses differently from what was really copied.
"""
import re

from lexicon import BY_LENGTH, WORDS, frequency_rank

# A callsign: one or two letters, optional digit, a digit, then letters.
CALL_RE = re.compile(r"^[A-Z0-9]{1,3}\d[A-Z]{1,4}(/[A-Z0-9]{1,3})?$")


def _wildcard_matches(token):
    """Lexicon words that fit the token, treating ? as any character."""
    if "?" not in token:
        return []
    out = []
    for w in BY_LENGTH.get(len(token), ()):
        if all(t == "?" or t == c for t, c in zip(token, w)):
            out.append(w)
    return sorted(out, key=frequency_rank)


def _near_matches(token, max_edits=1):
    """Words within one edit, for when a character was dropped or added.

    The decoder loses characters as well as mangling them: WARM came through as
    ?RM, which no same-length match can repair.
    """
    core = token.replace("?", "")
    if len(core) < 2:
        return []
    out = []
    for w in WORDS:
        if abs(len(w) - len(token)) > max_edits:
            continue
        if _edit(token, w) <= max_edits:
            out.append(w)
    return sorted(out, key=frequency_rank)


def _edit(a, b):
    """Edit distance where ? matches anything free of charge."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            same = (ca == cb) or ca == "?"
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if same else 1)))
        prev = cur
    return prev[-1]


def resolve_callsigns(tokens):
    """Reconcile the same callsign heard several times.

    Operators send their call two or three times precisely because one copy may
    not get through. That repetition is free redundancy: VA7LXX heard twice and
    VA7DXX once is almost certainly VA7LXX, and no dictionary is needed to say
    so.
    """
    # A ? could stand for a letter or a digit, and a callsign needs a digit in
    # the right place -- so try both before deciding a token is not a callsign.
    # Testing only one substitution made VA?LXX look like ordinary text, and
    # the whole consensus step quietly did nothing.
    def looks_like_call(t):
        return any(CALL_RE.match(t.replace("?", c)) for c in ("A", "0"))
    calls = [t for t in tokens if looks_like_call(t)]
    fixed = {}
    for i, a in enumerate(calls):
        if "?" not in a and len(a) >= 4:
            continue
        best, votes = None, 0
        for b in calls:
            if b is a or "?" in b:
                continue
            if len(b) == len(a) and _edit(a, b) <= 1:
                n = sum(1 for c in calls if c == b)
                if n > votes:
                    best, votes = b, n
        if best and votes >= 2:
            fixed[a] = best
    return fixed


# Pairs that occur constantly, used only to break a tie. ?RM DAY is WARM DAY,
# not QRM DAY -- both fit the pattern, and only one is something anyone says.
BIGRAMS = {
    ("WARM", "DAY"), ("COLD", "DAY"), ("NICE", "DAY"), ("GUD", "DAY"),
    ("GOOD", "MORNING"), ("GOOD", "EVENING"), ("GUD", "MORNING"),
    ("TNX", "FER"), ("TNX", "FOR"), ("HPE", "CUL"), ("VY", "73"),
    ("NAME", "IS"), ("QTH", "IS"), ("RIG", "IS"), ("ANT", "IS"),
    ("UR", "RST"), ("UR", "5NN"), ("HR", "IN"), ("PSE", "AGN"),
    ("SIG", "RPT"), ("MNI", "TNX"), ("SRI", "OM"), ("DR", "OM"),
}


def repair(text):
    """Return (tokens, marks) where marks say how each token was arrived at.

    'copied'  - read straight off the air
    'guessed' - filled from the vocabulary, one candidate fitted
    'unsure'  - several fitted, or it took an edit; shown but not to be trusted
    """
    tokens = text.split()
    call_fix = resolve_callsigns(tokens)
    out, marks = [], []
    for i, tok in enumerate(tokens):
        if tok in call_fix:
            out.append(call_fix[tok])
            marks.append("guessed")
            continue
        if "?" not in tok:
            out.append(tok)
            marks.append("copied")
            continue
        exact = _wildcard_matches(tok)
        near = [w for w in _near_matches(tok) if w not in exact]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        prv = out[-1] if out else ""

        # A following or preceding word can settle a tie that the pattern alone
        # cannot, and does it the way an operator does: by knowing what people
        # actually say.
        for cand in list(exact) + list(near):
            if (cand, nxt) in BIGRAMS or (prv, cand) in BIGRAMS:
                out.append(cand)
                marks.append("guessed" if cand in exact and len(exact) == 1
                             else "unsure")
                break
        else:
            if len(exact) == 1 and not near:
                out.append(exact[0]); marks.append("guessed")
            elif exact:
                # more than one reading fits; show the likeliest but say so
                out.append(exact[0]); marks.append("unsure")
            elif near:
                out.append(near[0]); marks.append("unsure")
            else:
                out.append(tok); marks.append("copied")
    return out, marks


def repair_text(text):
    tokens, marks = repair(text)
    return " ".join(tokens), marks
