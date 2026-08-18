"""Realistic on-air text to train and test against.

Training on random letters would spend the network's capacity learning that
letters are equally likely, which on the air they are not. Real CW is callsigns,
signal reports, and about thirty stock abbreviations, arranged in exchanges that
barely vary -- and a POTA activation is the most predictable of the lot. A model
that has seen ten thousand of those has a real prior on what comes next.
"""
import numpy as np

PREFIXES = ["K", "W", "N", "AA", "AB", "AC", "KB", "KC", "KD", "KE", "KF", "KG",
            "KI", "KJ", "KK", "KM", "KN", "KO", "NA", "NB", "NC", "ND", "NE",
            "WA", "WB", "WD", "VE", "VA", "G", "M", "DL", "F", "I", "EA", "SM",
            "OH", "JA", "JH", "VK", "ZL", "PY", "LU", "SP", "OK", "HA", "YB"]
SUFFIX = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STATES = ["CA", "OR", "WA", "NV", "AZ", "TX", "NM", "CO", "UT", "ID", "MT",
          "NY", "MA", "CT", "NH", "VT", "ME", "PA", "NJ", "MD", "VA", "NC",
          "SC", "GA", "FL", "AL", "TN", "KY", "OH", "MI", "IN", "IL", "WI",
          "MN", "IA", "MO", "AR", "LA", "OK", "KS", "NE", "SD", "ND", "WY"]
NAMES = ["JIM", "BOB", "TOM", "DAVE", "MIKE", "STEVE", "JOHN", "BILL", "RICK",
         "DAN", "PAUL", "GARY", "KEN", "LARRY", "ED", "AL", "RON", "PHIL"]
RIGS = ["FT991", "IC7300", "K3", "KX2", "KX3", "TS590", "FT818", "QMX"]


def callsign(rng):
    p = rng.choice(PREFIXES)
    digit = str(rng.integers(0, 10))
    n = int(rng.integers(1, 4))
    return p + digit + "".join(rng.choice(list(SUFFIX), n))


def rst(rng):
    """Reports are sent as 5NN far more often than the true number."""
    if rng.random() < 0.75:
        return "5NN"
    return f"5{rng.integers(5, 10)}{rng.integers(6, 10)}"


def pota_exchange(rng, me=None):
    """The activation exchange, which is most of what a POTA operator hears."""
    me = me or callsign(rng)
    him = callsign(rng)
    r = rng.random()
    if r < 0.30:
        return rng.choice([f"CQ POTA DE {me} {me} K",
                           f"CQ CQ POTA DE {me} K",
                           f"CQ DE {me} {me} K"])
    if r < 0.45:
        return him
    if r < 0.65:
        return rng.choice([f"{him} {rst(rng)} {rng.choice(STATES)}",
                           f"{him} GE {rst(rng)} {rst(rng)}"])
    if r < 0.80:
        return rng.choice([f"R {rst(rng)} {rng.choice(STATES)}",
                           f"RR TU {rst(rng)}", f"R R {rst(rng)} BK"])
    return rng.choice([f"TU {me} K", f"TU 73 DE {me} K", f"QSL TU DE {me} K",
                       f"73 GL DE {me}"])


def ragchew(rng):
    """A conversational QSO: longer, and much more varied vocabulary."""
    me, him = callsign(rng), callsign(rng)
    bits = [
        f"{him} DE {me}",
        f"GE OM TNX FER CALL",
        f"UR {rst(rng)} {rst(rng)} IN {rng.choice(STATES)}",
        f"NAME IS {rng.choice(NAMES)} {rng.choice(NAMES)}",
        f"RIG IS {rng.choice(RIGS)} PWR {rng.integers(5, 100)}W",
        f"WX SUNNY {rng.integers(30, 95)}F",
        f"ANT IS DIPOLE UP {rng.integers(20, 60)}FT",
        f"HW CPY? {him} DE {me} K",
        f"TNX FER QSO 73 ES GL",
    ]
    k = int(rng.integers(1, 4))
    start = int(rng.integers(0, len(bits) - k + 1))
    return " ".join(bits[start:start + k])


def sample(rng, kind=None):
    kind = kind or ("pota" if rng.random() < 0.65 else "ragchew")
    text = pota_exchange(rng) if kind == "pota" else ragchew(rng)
    return " ".join(text.upper().split())
