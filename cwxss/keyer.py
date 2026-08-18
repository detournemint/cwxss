"""Keyboard to CW: hand text to the radio's own keyer.

The radio generates the CW itself, over the same USB cable that carries CAT and
audio. That matters for the field: no keying interface to pack, no serial
handshake lines, and -- most importantly -- no dependence on a cheap laptop's
scheduler for timing. A laptop that stutters while redrawing a waterfall would
send a stuttering dah; the rig's keyer never does.

Requires a rig whose hamlib backend reports "Can send Morse: Y".
"""
import asyncio

# The radio's own keyer is a small buffer; overrun it and characters are lost.
# Sending in chunks and waiting for each keeps the buffer shallow, which also
# means stop() takes effect quickly rather than after everything queued.
CHUNK = 24
MAX_TEXT = 400
# hamlib error codes we can do something about
BUSBUSY = "-14"        # collision on the bus: another program spoke at once
BUSY_RETRIES = 4


class Keyer:
    def __init__(self, rig_cmd, log=print):
        self.rig_cmd = rig_cmd            # async fn: command string -> reply
        self.log = log
        self.sending = False
        self._abort = False

    async def capable(self):
        """Whether this rig can send morse at all.

        Matched loosely: hamlib separates the label and the answer with a tab,
        but that is a detail of its formatting rather than a promise.
        """
        caps = await self.rig_cmd("\\dump_caps")
        if not caps:
            return False
        for line in caps.splitlines():
            if "can send morse" in line.lower():
                return line.strip().lower().endswith("y")
        return False

    async def speed(self, wpm=None):
        """Get or set the rig's keyer speed."""
        if wpm is None:
            v = await self.rig_cmd("l KEYSPD")
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None
        wpm = max(4, min(60, int(wpm)))
        await self.rig_cmd(f"L KEYSPD {wpm}")
        return wpm

    async def raw(self, cmd, expect=0):
        """Send a CAT string straight to the radio through rigctld.

        rigctld's send_cmd is a pass-through: it does no interpretation, which
        is the point. Where hamlib's own idea of a rig is wrong, this is not.
        """
        return await self.rig_cmd(f"W {cmd} {expect}")

    async def break_in(self, on=True):
        """Turn break-in on, and confirm from the radio rather than from hamlib.

        hamlib reported break-in as enabled while the rig itself answered BI0 --
        off. Without it the rig will not key from CAT at all, so this is checked
        against the radio's own answer.
        """
        await self.raw(f"BI{1 if on else 0};", 0)
        await asyncio.sleep(0.3)
        reply = await self.raw("BI;", 4)
        state = (reply or "").strip()
        self.log(f"[keyer] break-in -> {state!r}")
        return state.startswith("BI1")

    async def send_ky(self, text):
        """Send CW using the rig's own keyer, bypassing hamlib's send_morse.

        hamlib asks the rig whether its keyer buffer is free before sending. The
        FT-991A answers that query with "?;" -- it does not implement it -- and
        hamlib reads the error as "busy" and refuses, permanently. Issuing KY
        ourselves sidesteps a check the radio was never able to answer.
        """
        reply = await self.raw(f"KY {text};", 0)
        ok = reply is not None and "RPRT 0" in (reply or "")
        self.log(f"[keyer] 'KY {text};' -> {reply!r}")
        return ok, reply

    async def send(self, text):
        """Send text as CW. Returns (ok, message).

        This transmits. It is called only from an explicit operator action --
        pressing enter on a typed line, or a macro key -- never on a timer and
        never as a side effect of anything else.
        """
        text = " ".join((text or "").upper().split())
        if not text:
            return False, "nothing to send"
        if len(text) > MAX_TEXT:
            return False, f"too long ({len(text)} characters, limit {MAX_TEXT})"
        if self.sending:
            return False, "already sending"

        self.sending, self._abort = True, False
        self.log(f"[keyer] send {text!r} at {await self.speed()} wpm")
        try:
            for i in range(0, len(text), CHUNK):
                if self._abort:
                    self.log("[keyer] aborted before chunk")
                    return False, "stopped"
                chunk = text[i:i + CHUNK]
                # A shared rig is normal: a logger or a digital-mode program may
                # be polling meters on the same serial line. Their reads are
                # short and ours can simply wait for a gap, rather than
                # declaring the radio broken.
                for attempt in range(BUSY_RETRIES):
                    reply = await self.rig_cmd(f"b {chunk}")
                    self.log(f"[keyer] 'b {chunk}' -> {reply!r}")
                    if not (reply and reply.strip().endswith(BUSBUSY)):
                        break
                    # Not a real collision on this rig: hamlib cannot ask the
                    # FT-991A whether its keyer is free, so it always concludes
                    # it is busy. Go straight to the radio instead.
                    self.log("[keyer] hamlib says busy; sending KY directly")
                    ok_raw, raw_reply = await self.send_ky(chunk)
                    if ok_raw:
                        reply = "RPRT 0"
                        break
                    await asyncio.sleep(0.4 * (attempt + 1))
                if reply is None:
                    return False, ("no answer from rigctld — is it still running?")
                if "RPRT 0" not in reply:
                    code = reply.strip().split()[-1] if reply.strip() else "?"
                    if code == BUSBUSY:
                        return False, ("the serial bus is busy — another program "
                                       "is polling the radio. Stop it, or give "
                                       "this one exclusive use of the rig.")
                    if code == "-4":
                        return False, "this rig's hamlib backend cannot send morse"
                    if code == "-11":
                        return False, ("not available — is the radio in CW mode?")
                    return False, (f"the rig refused it (RPRT {code})")
                ptt = await self.rig_cmd("t")
                self.log(f"[keyer] PTT during send -> {ptt!r}")
                w = await self.rig_cmd("\\wait_morse")
                self.log(f"[keyer] wait_morse -> {w!r}")
            return True, f"sent {len(text)} characters"
        finally:
            self.sending = False
            self.log("[keyer] send finished")

    async def stop(self):
        """Abort immediately, mid-character. The one control that must always
        work, so it does not wait for anything in flight."""
        self._abort = True
        reply = await self.rig_cmd("\\stop_morse")
        ok = reply is not None and "RPRT 0" in reply
        self.log(f"[keyer] stop -> {reply!r}")
        return ok


def estimate_seconds(text, wpm):
    """How long a message will take to send, for the operator's benefit."""
    from morse import PARIS_UNITS, to_units
    units = sum(n for _, n in to_units(text)) + 7
    return units * 60.0 / (PARIS_UNITS * max(wpm, 1))
