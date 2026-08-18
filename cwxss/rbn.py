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
