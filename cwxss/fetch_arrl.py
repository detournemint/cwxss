"""Build an evaluation set from ARRL code practice files.

These are the only readily available recordings of CW that come with an exact
transcript. They are machine-sent and studio-clean, so they are not training
data -- a model taught on them would learn that CW is easy -- but they are a
fair test of whether the decoder reads text it has never seen, at speeds we did
not choose, including the Farnsworth spacing ARRL uses at 15 wpm and below.

Downloaded for evaluation only. ARRL's material is not openly licensed, so
nothing fetched here is redistributed or trained on; the files stay local and
the repository ignores them.

    python3 cwxss/fetch_arrl.py --speeds 5,13,20,30
"""
import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = "http://www.arrl.org"
UA = "Mozilla/5.0 (X11; Linux x86_64) cwxss/evaluation"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def archive_links(speed):
    """(mp3, txt) pairs for one speed, newest first."""
    page = fetch(f"{BASE}/{speed}-wpm-code-archive").decode(errors="replace")
    mp3s = re.findall(r'href="(/files/file/Morse/Archive/[^"]+\.mp3)"', page)
    pairs = []
    for m in mp3s:
        # 260218_13WPM.mp3 sits beside 260218_13.txt
        t = re.sub(r"WPM\.mp3$", ".txt", m)
        pairs.append((m, t))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", default="5,13,20,30")
    ap.add_argument("--per-speed", type=int, default=1)
    ap.add_argument("--out", default="data/arrl")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for speed in [s.strip() for s in a.speeds.split(",") if s.strip()]:
        tag = speed.replace(".", "-")
        try:
            pairs = archive_links(tag)
        except Exception as e:
            print(f"  {speed} wpm: {type(e).__name__}: {e}")
            continue
        print(f"  {speed} wpm: {len(pairs)} files listed")
        for mp3, txt in pairs[:a.per_speed]:
            name = Path(mp3).stem
            wav = out / f"{name}.wav"
            if wav.exists():
                print(f"    {name}: already have it")
                continue
            try:
                raw = fetch(BASE + mp3)
                text = fetch(BASE + txt).decode(errors="replace")
            except Exception as e:
                print(f"    {name}: {type(e).__name__}")
                continue
            mp3_path = out / f"{name}.mp3"
            mp3_path.write_bytes(raw)
            (out / f"{name}.txt").write_text(text)
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i",
                                str(mp3_path), "-ac", "1", "-ar", "8000", str(wav)],
                               capture_output=True)
            mp3_path.unlink(missing_ok=True)
            if r.returncode == 0:
                print(f"    {name}: {wav.stat().st_size // 1024} KB, "
                      f"{len(text.split())} words of transcript")
            else:
                print(f"    {name}: ffmpeg failed")


if __name__ == "__main__":
    main()
