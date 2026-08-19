#!/usr/bin/env python3
"""Run everything.

    python3 tests/run.py

Needs only python3 and numpy. The browser test is skipped when no chromium is
installed rather than failing.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G, R, D, O = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def run(label, args):
    print(f"\n{D}$ {' '.join(str(a) for a in args)}{O}")
    try:
        code = subprocess.run(args, cwd=ROOT).returncode
    except FileNotFoundError:
        print(f"{D}  {label}: {args[0]} not installed, skipped{O}")
        return True
    ok = code == 0
    print(f"{G if ok else R}  {label}: {'passed' if ok else 'FAILED'}{O}")
    return ok


def check_page():
    """A syntax error in the inline script is a blank page for every operator,
    and no other test would notice."""
    html = (ROOT / "cwxss/static/index.html").read_text()
    blocks = re.findall(r"<script>([\s\S]*?)</script>", html)
    if not blocks:
        print(f"{R}  page: no script block found{O}")
        return False
    if not shutil.which("node"):
        print(f"{D}  page: node not installed, skipped{O}")
        return True
    for i, b in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(b)
            path = fh.name
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        Path(path).unlink(missing_ok=True)
        if r.returncode:
            print(f"{R}  page: block {i} has a syntax error{O}\n{r.stderr[:300]}")
            return False
    print(f"{G}  page: {len(blocks)} script block(s) parse{O}")
    return True


def check_setup():
    if not shutil.which("bash"):
        return True
    r = subprocess.run(["bash", "-n", "setup.sh"], cwd=ROOT, capture_output=True)
    ok = r.returncode == 0
    print(f"{G if ok else R}  setup.sh: {'parses' if ok else 'SYNTAX ERROR'}{O}")
    return ok


def main():
    results = [
        ("python", run("python", [sys.executable, "tests/test_cwxss.py"])),
        ("interface", run("interface",
                          [sys.executable, "tests/test_browser.py"])),
        ("page", check_page()),
        ("setup.sh", check_setup()),
    ]
    bad = [n for n, ok in results if not ok]
    print()
    if bad:
        print(f"{R}FAILED: {', '.join(bad)}{O}")
        return 1
    print(f"{G}All suites passed.{O}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
