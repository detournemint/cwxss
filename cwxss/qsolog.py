"""The log.

A park activation is ten contacts minimum and an ADIF file at the end of it, so
a decoder that cannot log is not a station. Everything here is written to disk
the moment it is entered: an activation ends when the battery does, and a log
held in memory is a log you can lose.

ADIF is the format every logging program and every awards programme reads. The
POTA fields are MY_SIG/MY_SIG_INFO for the park you are activating and
SIG/SIG_INFO for a park you are working -- park to park, which counts for both.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BANDS = [(1800, 2000, "160m"), (3500, 4000, "80m"), (5250, 5450, "60m"),
         (7000, 7300, "40m"), (10100, 10150, "30m"), (14000, 14350, "20m"),
         (18068, 18168, "17m"), (21000, 21450, "15m"), (24890, 24990, "12m"),
         (28000, 29700, "10m"), (50000, 54000, "6m"), (144000, 148000, "2m")]
# A callsign prefix always contains a letter. The old pattern allowed any
# alphanumerics before the digit, so "21C" validated -- and "SUNNY 21C" is a
# temperature that appears in real ragchew CW, which auto-logging duly offered
# to file as a contact. Every real prefix is one or two letters, a letter and a
# digit (A9), or a digit and a letter (4X, 9A).
CALL_RE = re.compile(
    r"^(?:[A-Z]{1,2}|[A-Z]\d|\d[A-Z])\d[A-Z]{1,4}(/[A-Z0-9]{1,4})?$")


# POTA references are a prefix, a dash and four or five digits: K-1234 in the
# US, VE-5082 in Canada, GB-0001 and so on.
PARK = re.compile(r"\b([A-Z0-9]{1,2}-\d{4,5})\b")
RST = re.compile(r"\b([1-5][1-9NA][1-9NA])\b")
STATE = re.compile(r"\b(A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|"
                   r"M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PA|RI|S[CD]|T[NX]|UT|"
                   r"V[AT]|W[AIVY])\b")
# The same rule as CALL_RE, for finding callsigns inside a line rather than
# validating one on its own. Keeping two different notions of what a callsign
# looks like is how "21C" got logged.
CALL = re.compile(
    r"\b((?:[A-Z]{1,2}|[A-Z]\d|\d[A-Z])\d[A-Z]{1,4}(?:/[A-Z0-9]{1,4})?)\b")


def read_exchange(text, my_call="", their_call=""):
    """Pull a contact out of decoded CW.

    A park activation is ten contacts and every one of them has to be typed
    while the next station is already calling. The exchange is short and highly
    conventional, so most of it can be read straight off the transcript:

        K6XSS DE W1ABC BK
        W1ABC 599 CA
        R 599 TX TU
        K-1234

    Returns what could be found and how sure it is, rather than guessing.
    Nothing here logs anything: a wrong callsign in a POTA upload is worse than
    a missing one, so the caller decides what to do with a low score.

    Callsigns are taken by consensus. Off-air CW mangles them -- KK6IK arrives
    as KKK6IK -- but a station sends its own call several times in an exchange,
    and the variants disagree in different places. `guess.resolve_callsigns`
    already does that work.
    """
    import guess
    up = (text or "").upper()
    mine = (my_call or "").upper()
    tokens = up.split()

    fixed = guess.resolve_callsigns(tokens)
    tokens = [fixed.get(t, t) for t in tokens]

    calls = []
    for t in tokens:
        m = CALL.fullmatch(t.strip(".,?=/+-") if "/" not in t else t)
        if not m:
            continue
        c = m.group(1)
        if mine and (c == mine or c.startswith(mine + "/")):
            continue        # our own call is not the contact
        calls.append(c)

    call = (their_call or "").upper()
    score = 1.0 if call else 0.0
    if not call and calls:
        # The station worked is the callsign that appears most; ties go to the
        # later one, because a QSO ends with the other operator's call.
        counts = {}
        for i, c in enumerate(calls):
            counts[c] = counts.get(c, (0, 0))
            counts[c] = (counts[c][0] + 1, i)
        call, (n, _) = max(counts.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        score = min(1.0, n / 2.0)      # seen twice or more is a good sign

    park = PARK.search(up)
    rst = RST.findall(up)

    # The state is taken only where the exchange puts it -- straight after the
    # report, "599 CA" -- and never by scanning the text for anything that
    # looks like one. Half the state abbreviations are ordinary CW: DE means
    # "from" and would otherwise log every contact as Delaware, HI is laughter,
    # IN OR ME OH are words, AR and SK are prosigns. Position disambiguates
    # them; spelling cannot.
    state = ""
    for i, t in enumerate(tokens[:-1]):
        if RST.fullmatch(t.strip(".,")):
            nxt = tokens[i + 1].strip(".,")
            if STATE.fullmatch(nxt):
                state = nxt
    return {
        "call": call,
        "rst_rcvd": (rst[-1].replace("N", "9").replace("A", "1")
                     if rst else "599"),
        "their_park": park.group(1) if park else "",
        "state": state,
        "confidence": round(score, 2),
        "candidates": sorted(set(calls)),
    }


def band_of(hz):
    khz = (hz or 0) / 1000.0
    for lo, hi, name in BANDS:
        if lo <= khz <= hi:
            return name
    return ""


def valid_call(call):
    return bool(CALL_RE.match((call or "").upper().strip()))


class Log:
    def __init__(self, path=None):
        self.path = Path(path or (Path.home() / "cw-log.jsonl"))
        self.qsos = []
        self._load()

    def _load(self):
        try:
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.qsos.append(json.loads(line))
        except (OSError, ValueError):
            pass

    def worked(self, call, band=None, same_day=True):
        """Previous contacts with this station.

        A dupe on POTA is the same station on the same band on the same day; the
        next day, or another band, counts again. Hunters chase activators across
        bands all afternoon, so this has to be right or it will refuse contacts
        that count.
        """
        call = (call or "").upper().strip()
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = []
        for q in self.qsos:
            if q.get("call", "").upper() != call:
                continue
            if band and q.get("band") != band:
                continue
            if same_day and q.get("date") != today:
                continue
            out.append(q)
        return out

    def add(self, call, freq_hz=None, rst_sent="599", rst_rcvd="599",
            mode="CW", wpm=None, their_park="", my_park="", state="",
            comment=""):
        call = (call or "").upper().strip()
        if not valid_call(call):
            return None, f"{call!r} does not look like a callsign"
        now = datetime.now(timezone.utc)
        q = {
            "call": call,
            "date": now.strftime("%Y%m%d"),
            "time": now.strftime("%H%M%S"),
            "freq_mhz": round((freq_hz or 0) / 1e6, 6) or None,
            "band": band_of(freq_hz),
            "mode": mode,
            "rst_sent": rst_sent, "rst_rcvd": rst_rcvd,
            "wpm": wpm, "their_park": their_park, "my_park": my_park,
            "state": state, "comment": comment,
        }
        self.qsos.append(q)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(q) + "\n")
        except OSError as e:
            return q, f"logged, but could not write the file: {e}"
        return q, ""

    def remove_last(self):
        """Undo. Fat fingers happen at 25 wpm."""
        if not self.qsos:
            return None
        q = self.qsos.pop()
        try:
            self.path.write_text(
                "".join(json.dumps(x) + "\n" for x in self.qsos))
        except OSError:
            pass
        return q

    def today(self):
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        return [q for q in self.qsos if q.get("date") == d]

    def summary(self):
        today = self.today()
        bands = {}
        for q in today:
            bands[q.get("band", "?")] = bands.get(q.get("band", "?"), 0) + 1
        return {
            "total": len(self.qsos),
            "today": len(today),
            "bands": bands,
            # POTA counts an activation at ten contacts.
            "activated": len(today) >= 10,
            "needed": max(0, 10 - len(today)),
        }


def adif(qsos, station_call="", my_park=""):
    """An ADIF file, which is what every logger and awards programme reads."""
    def f(name, value):
        value = "" if value is None else str(value)
        return f"<{name.upper()}:{len(value)}>{value}" if value else ""

    out = [f"ADIF export from cwxss  {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
           "<ADIF_VER:5>3.1.0", "<PROGRAMID:5>cwxss", "<EOH>", ""]
    for q in qsos:
        row = "".join([
            f("call", q.get("call")),
            f("qso_date", q.get("date")),
            f("time_on", q.get("time")),
            f("band", q.get("band")),
            f("freq", q.get("freq_mhz")),
            f("mode", q.get("mode", "CW")),
            f("rst_sent", q.get("rst_sent")),
            f("rst_rcvd", q.get("rst_rcvd")),
            f("state", q.get("state")),
            f("station_callsign", station_call),
            f("operator", station_call),
            # POTA: MY_SIG is the park being activated, SIG the one worked.
            f("my_sig", "POTA" if (q.get("my_park") or my_park) else ""),
            f("my_sig_info", q.get("my_park") or my_park),
            f("sig", "POTA" if q.get("their_park") else ""),
            f("sig_info", q.get("their_park")),
            f("comment", q.get("comment")),
        ])
        out.append(row + "<EOR>")
    return "\n".join(out) + "\n"
