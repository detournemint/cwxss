"""Who is hearing us, from the Reverse Beacon Network.

Skimmer stations around the world decode CW continuously and report what they
hear. For CW this does the job PSK Reporter does for FT8, with one difference
worth knowing before you wonder why you are not listed: skimmers report stations
calling CQ. Work someone without ever calling CQ and nothing will spot you, however
well you are getting out.

The feed is a live stream in DX cluster format, so this holds the connection
open rather than polling, and spots arrive within seconds of transmitting.
"""
import asyncio
import re
import time

HOST, PORT = "telnet.reversebeacon.net", 7000

# DX de PI4CC-#:   7010.40  9A2V     CW    31 dB  22 WPM  CQ      0413Z
SPOT = re.compile(
    r"DX de\s+(?P<spotter>[\w/\-#]+):\s+(?P<khz>[\d.]+)\s+(?P<dx>[\w/]+)\s+"
    r"(?P<mode>\w+)\s+(?P<snr>-?\d+)\s*dB\s+(?P<wpm>\d+)\s*WPM\s+(?P<kind>\w+)")


# Skimmers whose propagation resembles ours closely enough to be worth
# following. A spot from Europe says a station is transmitting; it says nothing
# about whether this antenna can hear it, and 20m to Europe and 20m to
# California are different paths at the same moment. Override for a station
# outside the western US.
NEARBY = ("W6", "K6", "N6", "AI6", "KM6", "KE6", "AE6", "WW6",
          "W7", "K7", "N7", "AA7", "KE7", "AI7", "VE7", "KH6")


def active_frequencies(call, seconds=60, prefixes=NEARBY, min_snr=3,
                       bands=None, log=print):
    """Where nearby skimmers are hearing CW right now, strongest first.

    A blind band scan is a lottery. It pauses four seconds on each step and
    comes back about every two and a half minutes, while a station calling CQ
    transmits maybe a third of the time -- so most passes miss most stations,
    and an empty sweep mostly measures the sampling rather than the band. Three
    separate sweeps here reported nothing on 20m while skimmers forty miles
    away were spotting stations inside the range being swept.

    Skimmers are already listening on every frequency at once. Asking them
    where to point the radio turns a search into a lookup.

    Only CQ spots are returned, because a station calling CQ will still be
    calling in a minute and a station answering one will not.
    """
    import socket
    out = {}
    try:
        sock = socket.create_connection((HOST, PORT), timeout=20)
        sock.sendall((call.upper() + "\n").encode())
        end = time.time() + seconds
        buf = b""
        while time.time() < end:
            try:
                sock.settimeout(3)
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                m = SPOT.search(line.decode(errors="replace"))
                if not m:
                    continue
                d = m.groupdict()
                if d["mode"].upper() != "CW" or d["kind"].upper() != "CQ":
                    continue
                if not d["spotter"].rstrip("-#").upper().startswith(
                        tuple(prefixes)):
                    continue
                snr = int(d["snr"])
                if snr < min_snr:
                    continue
                khz = float(d["khz"])
                if bands and not any(lo <= khz * 1000 <= hi
                                     for lo, hi in bands):
                    continue
                # Same frequency spotted by several skimmers: keep the best
                # report, it is the same signal.
                prev = out.get(round(khz, 1))
                if prev is None or snr > prev["snr"]:
                    out[round(khz, 1)] = {
                        "khz": khz, "dx": d["dx"], "snr": snr,
                        "wpm": int(d["wpm"]), "spotter":
                        d["spotter"].rstrip("-#"),
                    }
        sock.close()
    except OSError as e:
        log(f"  rbn unavailable: {e}")
        return []
    return sorted(out.values(), key=lambda s: -s["snr"])


class RbnMonitor:
    """Watches the feed for our own callsign."""

    def __init__(self, call, log=print, keep=60):
        self.call = (call or "").upper()
        self.log = log
        self.keep = keep
        self.spots = []          # newest last
        self.connected = False
        self.last_error = ""
        self.seen_total = 0      # spots of any station, to show the feed is alive

    def state(self):
        now = time.time()
        recent = [s for s in self.spots if now - s["at"] < 3600]
        return {
            "connected": self.connected,
            "error": self.last_error,
            "call": self.call,
            "count": len(recent),
            "best": max((s["snr"] for s in recent), default=None),
            "feed_rate": self.seen_total,
            "spots": [
                {k: s[k] for k in ("spotter", "khz", "snr", "wpm", "kind", "ago")}
                for s in sorted(recent, key=lambda s: -s["at"])[:12]
            ],
        }

    def _add(self, m):
        now = time.time()
        self.spots.append({
            "spotter": m["spotter"].rstrip("-#"),
            "khz": float(m["khz"]), "snr": int(m["snr"]),
            "wpm": int(m["wpm"]), "kind": m["kind"], "at": now, "ago": 0,
        })
        self.spots = self.spots[-self.keep:]
        for s in self.spots:
            s["ago"] = int(now - s["at"])
        self.log(f"[rbn] {self.call} heard by {m['spotter']} on {m['khz']} "
                 f"at {m['snr']} dB, {m['wpm']} wpm")

    async def run(self, on_spot=None):
        """Stay connected, reconnecting when the feed drops."""
        while True:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(HOST, PORT), 20)
                await asyncio.wait_for(reader.read(256), 15)   # "enter your call"
                writer.write((self.call + "\n").encode())
                await writer.drain()
                self.connected, self.last_error = True, ""
                self.log(f"[rbn] connected as {self.call}")
                while True:
                    line = await asyncio.wait_for(reader.readline(), 300)
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    m = SPOT.search(text)
                    if not m:
                        continue
                    self.seen_total += 1
                    if m["dx"].upper() == self.call:
                        self._add(m)
                        if on_spot:
                            await on_spot(self.state())
            except Exception as e:
                self.last_error = f"{type(e).__name__}"
                self.log(f"[rbn] disconnected: {type(e).__name__}: {e}")
            self.connected = False
            await asyncio.sleep(15)
