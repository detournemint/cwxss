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
CALL_RE = re.compile(r"^[A-Z0-9]{1,3}\d[A-Z]{1,4}(/[A-Z0-9]{1,4})?$")


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
