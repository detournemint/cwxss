"""cwxss - a CW decoder and keyboard keyer you run from a browser.

Built for a cheap laptop in a park: no GPU, modest CPU, and a web interface so
the operating position can be a phone or tablet propped against the rig.

    python3 cwxss/server.py --demo            # synthetic signal, no radio
    python3 cwxss/server.py --device plughw:2,0 --rig 127.0.0.1:4532
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture, config, dtrkey, hamtext, keyer as keyer_mod  # noqa: E402
import rbn as rbn_mod                                           # noqa: E402
import stream, synth                                            # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


class State:
    def __init__(self):
        self.decoder = stream.StreamDecoder()
        self.clients = set()
        self.source = "none"
        self.keyer = None
        self.wpm = 20
        self.sending = ""
        self.truth = ""            # in demo mode, what was actually sent
        self.dtr = None            # keying by control line, when available
        self.rbn = None            # who is hearing us, from the skimmer network
        self.cfg = config.load()
        self.his = ""              # the station being worked, for {his}
        # A rolling buffer of what we have just heard. Real off-air CW is worth
        # keeping the moment it appears -- by the time you have decided a signal
        # was interesting it has usually finished sending.
        self.recent = np.zeros(0, dtype=np.float32)
        self.recent_s = 180.0

    async def broadcast(self, kind, data):
        if not self.clients:
            return
        msg = json.dumps({"type": kind, "data": data})
        for ws in list(self.clients):
            try:
                await ws.send_str(msg)
            except Exception:
                self.clients.discard(ws)


ST = State()


async def rig_cmd_bound(cmd):        # replaced at startup when --rig is given
    return None


async def rig_cmd(cmd, host="127.0.0.1", port=4532, timeout=6.0):
    """One command to rigctld. None if it cannot be reached."""
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except Exception:
        return None
    try:
        w.write((cmd + "\n").encode())
        await w.drain()
        # Read until the reply stops arriving, not just the first 4 KB.
        # dump_caps is about 10 KB, and "Can send Morse" sits well past the
        # first block -- truncating it made a perfectly capable rig look as
        # though it could not key at all.
        chunks = []
        while True:
            try:
                part = await asyncio.wait_for(r.read(65536), 0.6)
            except asyncio.TimeoutError:
                break
            if not part:
                break
            chunks.append(part)
            if len(part) < 65536 and len(chunks) > 0 and b"RPRT" in part:
                break
        return b"".join(chunks).decode(errors="replace").strip()
    except Exception:
        return None
    finally:
        w.close()


async def demo_source():
    """A synthetic band: one station after another, with varying conditions.

    So the interface can be seen working without a radio, and so the decoder can
    be watched against text we know -- the display shows both.
    """
    ST.source = "demo (synthetic signal, no radio)"
    rng = np.random.default_rng()
    while True:
        text = hamtext.sample(rng)
        wpm = float(rng.uniform(14, 28))
        r = rng.random()
        fist = (synth.Fist.keyer(rng) if r < 0.35 else
                synth.Fist.good_op(rng) if r < 0.75 else synth.Fist.rough_op(rng))
        audio = synth.render(text, wpm=wpm, pitch=float(rng.uniform(500, 750)),
                             snr_db=float(rng.uniform(6, 26)), fist=fist,
                             qsb_depth=0.5 if rng.random() < 0.3 else 0.0)
        ST.truth = text
        await ST.broadcast("truth", {"text": text, "wpm": round(wpm, 1)})
        chunk = int(0.2 * synth.DEFAULT_RATE)
        for i in range(0, len(audio), chunk):
            ST.decoder.feed(audio[i:i + chunk])
            await ST.broadcast("decode", ST.decoder.state())
            await asyncio.sleep(0.2)          # play it at real speed
        await asyncio.sleep(1.0)


async def audio_source(device, rate=8000):
    """Live audio from the radio, straight out of arecord."""
    ST.source = f"live: {device}"
    proc = await asyncio.create_subprocess_exec(
        "arecord", "-D", device, "-f", "S16_LE", "-r", str(rate), "-c", "1",
        "-t", "raw", "-q",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    chunk = int(0.2 * rate) * 2                 # bytes: 16-bit mono
    try:
        while True:
            raw = await proc.stdout.readexactly(chunk)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            ST.recent = np.concatenate([ST.recent, samples])[
                -int(ST.recent_s * rate):]
            ST.decoder.feed(samples)
            await ST.broadcast("decode", ST.decoder.state())
    except (asyncio.IncompleteReadError, asyncio.CancelledError):
        pass
    finally:
        if proc.returncode is None:
            proc.kill()


async def watch_transmission(rig, seconds):
    """Sample PTT and power while we key, so a send that produced no RF is
    distinguishable from one nobody happened to hear.

    "Sent 59 elements" only means the software toggled a line 59 times. Whether
    the radio keyed, and whether any power came out, is a different question --
    and it is the one that matters when the skimmers stay quiet.
    """
    keyed, watts = 0, 0.0
    end = asyncio.get_event_loop().time() + seconds
    while asyncio.get_event_loop().time() < end:
        ptt = await rig("t")
        if (ptt or "").strip() == "1":
            keyed += 1
            w = await rig("l RFPOWER_METER_WATTS")
            try:
                watts = max(watts, float(w))
            except (TypeError, ValueError):
                pass
        await asyncio.sleep(0.25)
    return keyed, watts


async def ws_handler(req):
    ws = web.WebSocketResponse(heartbeat=25)
    await ws.prepare(req)
    ST.clients.add(ws)
    await ws.send_str(json.dumps({"type": "hello", "data": {
        "source": ST.source, "wpm": ST.wpm, "truth": ST.truth,
        "can_send": (ST.dtr is not None or ST.keyer is not None),
        "keying": ("control line" if ST.dtr else "hamlib" if ST.keyer else "none"),
        "cfg": ST.cfg, "his": ST.his,
        "rbn": ST.rbn.state() if ST.rbn else None,
        "build": build_id(),
        "decode": ST.decoder.state()}}))
    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                req_msg = json.loads(msg.data)
            except ValueError:
                continue
            act = req_msg.get("action")

            if act == "macro":
                # Macros are stored with placeholders and filled in at the
                # moment of sending, so changing the callsign being worked does
                # not mean editing ten messages.
                i = req_msg.get("index", -1)
                msgs = config.messages(ST.cfg)
                text = (config.expand(msgs[i]["text"], ST.cfg, ST.his)
                        if 0 <= i < len(msgs) else "")
                if not (ST.dtr or ST.keyer):
                    ok, who = False, "no radio connected — nothing to key"
                elif not text:
                    ok, who = False, "that macro is empty"
                else:
                    ST.sending = text
                    await ST.broadcast("sending", {"text": text})
                    from morse import PARIS_UNITS, to_units
                    secs = (sum(n for _, n in to_units(text)) * 60.0
                            / (PARIS_UNITS * max(ST.wpm, 1)))
                    watch = asyncio.create_task(
                        watch_transmission(rig_cmd_bound, secs + 1))
                    ok, who = (await ST.dtr.send(text, ST.wpm) if ST.dtr
                               else await ST.keyer.send(text))
                    keyed, watts = await watch
                    ST.sending = ""
                    await ST.broadcast("sending", {"text": ""})
                    if ok and keyed == 0:
                        ok = False
                        who = ("the key line moved but the radio never keyed — "
                               "check menu 060 PC KEYING = DTR")
                    elif ok:
                        who = f"{who}, {watts:.0f} W out"
                print(f"[send] macro {i} {text!r} -> ok={ok} {who} "
                      f"(ptt seen {keyed if 'keyed' in dir() else '?'})", flush=True)
                await ws.send_str(json.dumps({"type": "ack", "data": {
                    "action": "macro", "ok": ok, "who": who or text}}))
            elif act == "save":
                # Keep what is in the buffer, with everything we know about it.
                # A recording without its frequency, mode and speed is close to
                # useless later.
                secs = float(req_msg.get("seconds") or 60)
                rate = 8000
                audio = ST.recent[-int(secs * rate):]
                if audio.size < rate:
                    ok, who = False, "nothing buffered yet"
                else:
                    from datetime import datetime, timezone
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    out = capture.DEFAULT_DIR / f"cw-{stamp}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    import wave
                    with wave.open(str(out), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(rate)
                        w.writeframes((np.clip(audio, -1, 1) * 32767
                                       ).astype(np.int16).tobytes())
                    st = ST.decoder.state()
                    capture.write_sidecar(
                        out, freq=await rig_cmd("f"), mode=await rig_cmd("m"),
                        pitch_hz=st.get("pitch"), wpm=st.get("wpm"),
                        snr_db=st.get("snr"), decoded=st.get("text", "")[-500:],
                        note=req_msg.get("note", ""), seconds=round(len(audio)/rate, 1))
                    ok, who = True, f"saved {out.name} ({len(audio)/rate:.0f}s)"
                await ws.send_str(json.dumps({"type": "ack", "data": {
                    "action": "save", "ok": ok, "who": who}}))
            elif act == "his":
                ST.his = (req_msg.get("call") or "").strip().upper()
                await ST.broadcast("his", {"call": ST.his})
            elif act == "config":
                ST.cfg.update(req_msg.get("cfg") or {})
                config.save(ST.cfg)
                await ST.broadcast("cfg", ST.cfg)
            elif act == "macroset":
                want = req_msg.get("set")
                if want in ST.cfg.get("sets", {}):
                    ST.cfg["active"] = want
                    config.save(ST.cfg)
                    await ST.broadcast("cfg", ST.cfg)
            elif act == "send":
                text = (req_msg.get("text") or "").strip()
                if not (ST.dtr or ST.keyer):
                    ok, who = False, "no radio connected — nothing to key"
                else:
                    ST.sending = text
                    await ST.broadcast("sending", {"text": text})
                    ok, who = (await ST.dtr.send(text, ST.wpm) if ST.dtr
                               else await ST.keyer.send(text))
                    ST.sending = ""
                    await ST.broadcast("sending", {"text": ""})
                print(f"[send] {text!r} -> ok={ok} {who}", flush=True)
                await ws.send_str(json.dumps({"type": "ack", "data": {
                    "action": "send", "ok": ok, "who": who}}))
            elif act == "keytest":
                # Prove the line moves before anything goes on the air. Harmless
                # while the radio's PC KEYING menu item is off.
                ok, who = ((await ST.dtr.selftest()) if ST.dtr
                           else (False, "no key line configured"))
                await ws.send_str(json.dumps({"type": "ack", "data": {
                    "action": "keytest", "ok": ok, "who": who}}))
            elif act == "stop":
                ok = False
                if ST.dtr:
                    ok = await ST.dtr.stop()
                if ST.keyer:
                    ok = await ST.keyer.stop() or ok
                ST.sending = ""
                await ST.broadcast("sending", {"text": ""})
                await ws.send_str(json.dumps({"type": "ack", "data": {
                    "action": "stop", "ok": ok, "who": "stopped"}}))
            elif act == "wpm":
                try:
                    ST.wpm = max(5, min(50, int(req_msg.get("wpm", 20))))
                except (TypeError, ValueError):
                    pass
                if ST.keyer:
                    await ST.keyer.speed(ST.wpm)
                await ST.broadcast("wpm", {"wpm": ST.wpm})
            elif act == "clear":
                ST.decoder.committed = ""
                await ST.broadcast("decode", ST.decoder.state())
    finally:
        ST.clients.discard(ws)
    return ws


def build_id():
    """Identifies the interface on disk.

    A browser goes on running the JavaScript it loaded, so deploying a fix does
    nothing for a page already open -- it keeps calling the old endpoints and
    behaving the old way, which is indistinguishable from the fix not working.
    """
    try:
        st = (STATIC / "index.html").stat()
        return f"{int(st.st_mtime)}-{st.st_size}"
    except OSError:
        return "unknown"


async def index(req):
    """Serve the interface, and tell the browser not to keep it.

    The page is the whole application -- markup, styles and script in one file.
    Cached, it goes on running last week's code against this week's server,
    which is indistinguishable from the server being broken and costs an
    operator a hard refresh they have no reason to know they need. It is a few
    tens of kilobytes over a LAN; revalidating it every time is free.
    """
    resp = web.FileResponse(STATIC / "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@web.middleware
async def no_cache_static(request, handler):
    """Same for anything else we serve: this is a local tool, not a CDN."""
    resp = await handler(request)
    if request.path.startswith("/static/") or request.path == "/":
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic signal, no radio")
    ap.add_argument("--device", default=None, help="ALSA capture device")
    ap.add_argument("--rig", default=None, help="rigctld host:port")
    ap.add_argument("--keyline", default=None,
                    help="serial port for CW keying, e.g. /dev/ttyUSB1")
    ap.add_argument("--keyline-signal", default="dtr", choices=("dtr", "rts"))
    ap.add_argument("--port", type=int, default=8074)
    ap.add_argument("--bind", default="0.0.0.0")
    a = ap.parse_args()

    # Keying by control line, the way flrig and fldigi do it. Hamlib's
    # send_morse cannot work on every rig -- on an FT-991A the CAT KY command
    # only replays stored keyer memories -- so the line takes precedence when
    # one is configured.
    if a.keyline:
        dk = dtrkey.DtrKeyer(a.keyline, a.keyline_signal,
                             log=lambda m: print(m, flush=True))
        ok, why = dk.available()
        print(f"  keyline: {why}")
        if ok:
            ST.dtr = dk

    if a.rig:
        host, _, port = a.rig.partition(":")
        async def cmd(c):
            return await rig_cmd(c, host, int(port or 4532))
        globals()["rig_cmd_bound"] = cmd
        k = keyer_mod.Keyer(cmd, log=lambda m: print(m, flush=True))
        if await k.capable():
            ST.keyer = k
            ST.wpm = await k.speed() or 20
            # The rig will not key from CAT with break-in off, and hamlib's
            # answer about it cannot be trusted -- ask the radio.
            bk = await k.break_in(True)
            print(f"  keying enabled via {a.rig}, {ST.wpm} wpm, "
                  f"break-in {'on' if bk else 'COULD NOT ENABLE'}")
        else:
            print(f"  {a.rig}: this rig cannot send morse — decode only")

    app = web.Application(middlewares=[no_cache_static])
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", STATIC)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, a.bind, a.port).start()

    call = ST.cfg.get("call", "")
    if call and call.upper() != "N0CALL":
        ST.rbn = rbn_mod.RbnMonitor(call, log=lambda m: print(m, flush=True))

        async def on_spot(state):
            await ST.broadcast("rbn", state)
        asyncio.create_task(ST.rbn.run(on_spot))
        print(f"  watching the Reverse Beacon Network for {call}")

    if a.demo:
        asyncio.create_task(demo_source())
    elif a.device:
        asyncio.create_task(audio_source(a.device))
    else:
        dev = capture.pick_device()
        if dev:
            asyncio.create_task(audio_source(dev["alsa"]))
        else:
            print("  no capture device found; run with --demo to see the interface")
    print(f"  http://{a.bind}:{a.port}   source: {ST.source}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
