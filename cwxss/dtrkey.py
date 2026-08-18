"""Key the transmitter by toggling a serial control line.

This is how the established CW software does it -- flrig, fldigi, N1MM -- and
having tried the alternative, the reason is clear. Hamlib's `send_morse` hands
text to the radio's own keyer, which on a Yaesu means the CAT `KY` command; but
on the FT-991A KY only replays keyer memories 1-4, it cannot accept arbitrary
text. Hamlib asks the rig whether the keyer is free, the rig answers "?;"
because it does not implement the query, and hamlib reports the radio busy
forever.

Toggling a control line sidesteps all of that: it is the electrical equivalent
of a straight key, and every rig with a keying input understands it. The cost is
that the timing becomes ours to generate rather than the radio's, which is why
morse.py defines it exactly.

The radio needs its PC KEYING menu item set to DTR (item 060 on an FT-991A),
and the line used here must be the keying port, not the CAT port.
"""
import asyncio
import time

from morse import dit_seconds, to_units

# Sending happens in a worker thread. Sleeping on the event loop would make the
# timing hostage to whatever else the server is doing -- a decode pass landing
# mid-character would stretch a dit into a dah.
MAX_TEXT = 400


class DtrKeyer:
    def __init__(self, port="/dev/ttyUSB1", line="dtr", log=print):
        self.port = port
        self.line = line            # "dtr" or "rts", per the radio's menu
        self.log = log
        self.sending = False
        self._abort = False
        self._serial = None

    def available(self):
        try:
            import serial          # noqa: F401
        except ImportError:
            return False, "pyserial is not installed"
        from pathlib import Path
        if not Path(self.port).exists():
            return False, f"{self.port} does not exist"
        return True, f"keying {self.port} via {self.line.upper()}"

    def _open(self):
        import serial
        if self._serial is None or not self._serial.is_open:
            # A keying line needs no baud rate agreement: nothing is being
            # transmitted as data. Only the control line matters.
            self._serial = serial.Serial(self.port, 9600, timeout=0)
            self._key(False)
        return self._serial

    def _key(self, down):
        if self.line == "rts":
            self._serial.rts = bool(down)
        else:
            self._serial.dtr = bool(down)

    def _send_blocking(self, text, wpm):
        """Play the message out on the keying line. Runs in a worker thread."""
        unit = dit_seconds(wpm)
        seq = to_units(text)
        self._open()
        sent = 0
        try:
            for on, units in seq:
                if self._abort:
                    break
                self._key(on)
                # Sleep against a deadline rather than for a duration, so that
                # scheduling jitter does not accumulate over a long message.
                target = time.perf_counter() + units * unit
                while True:
                    remaining = target - time.perf_counter()
                    if remaining <= 0 or self._abort:
                        break
                    time.sleep(min(remaining, 0.002))
                if on:
                    sent += 1
        finally:
            self._key(False)        # never leave the transmitter keyed
        return sent

    async def send(self, text, wpm=20):
        """Send text as CW. Transmits; called only from an operator action."""
        text = " ".join((text or "").upper().split())
        if not text:
            return False, "nothing to send"
        if len(text) > MAX_TEXT:
            return False, f"too long ({len(text)} characters, limit {MAX_TEXT})"
        ok, why = self.available()
        if not ok:
            return False, why
        if self.sending:
            return False, "already sending"

        self.sending, self._abort = True, False
        self.log(f"[dtr] send {text!r} at {wpm} wpm on {self.port}/{self.line}")
        try:
            n = await asyncio.get_event_loop().run_in_executor(
                None, self._send_blocking, text, wpm)
            if self._abort:
                return False, "stopped"
            return True, f"sent {len(text)} characters, {n} elements"
        except Exception as e:
            self.log(f"[dtr] failed: {type(e).__name__}: {e}")
            return False, f"{type(e).__name__}: {e}"
        finally:
            self.sending = False

    async def stop(self):
        """Drop the key immediately. Must work whatever else is happening."""
        self._abort = True
        try:
            if self._serial is not None and self._serial.is_open:
                self._key(False)
        except Exception:
            pass
        self.log("[dtr] stop")
        return True

    async def selftest(self, seconds=2.0):
        """Toggle the line without sending anything meaningful.

        Harmless when the radio's PC KEYING menu item is off, which makes it a
        safe way to prove the port opens and the line moves before anything is
        put on the air.
        """
        ok, why = self.available()
        if not ok:
            return False, why
        try:
            self._open()
            for _ in range(int(seconds * 2)):
                self._key(True); time.sleep(0.05)
                self._key(False); time.sleep(0.2)
            return True, f"toggled {self.line.upper()} on {self.port}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
