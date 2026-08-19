"""End-to-end tests for the interface, in a real browser.

Everything else in this suite tests functions. This one loads the page the
station actually serves, lets it render, and drives real events at it -- the
only way to catch a button wired to the wrong handler, a render path that
throws on a field it did not expect, or a script that fails on load and leaves
the operator a blank page.

There is no radio and no server here. The page's WebSocket is replaced before
its own script runs, so it receives a scripted conversation instead of a live
one and every message it tries to send is captured.

    python3 tests/test_browser.py

Skipped, not failed, when no chromium is installed.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "cwxss/static/index.html"
CHROME = next((c for c in ("chromium", "chromium-browser", "google-chrome",
                           "google-chrome-stable")
               if shutil.which(c)), None)

# Installed before the page's script, so `new WebSocket` is ours by the time
# connect() runs. Delivers a hello and one decode, then stands still.
STUB = r"""
<script>
window.__sent = [];
window.__err  = null;
window.addEventListener("error", e => { window.__err = String(e.message); });
const NOW = Math.floor(Date.now() / 1000);
window.__state = {
  hello: {source: "test", can_send: true, wpm: 22, build: "test",
          cfg: {call: "K6XSS", park: "K-1234", rst: "599", macroset: "pota"},
          his: "", decode: null, rbn: null, log: null},
  decode: {
    pitch: 700, wpm: 22, snr: 14, bandwidth: 70, quiet: "",
    envelope: new Array(80).fill(0.2),
    text: "CQ POTA DE K6XSS K",
    pending: "TU",
    history: [
      {at: NOW - 600, text: "CQ POTA DE K6XSS K"},
      {at: NOW - 300, text: "K6XSS DE W1ABC BK"},
      {at: NOW - 10,  text: "W1ABC 599 CA R 599 TX TU"}
    ],
    guessed: [
      {w: "WX",      m: "copied"},
      {w: "WEATNER", m: "struck"},
      {w: "EEEEEE",  m: "cancel"},
      {w: "WEATHER", m: "copied"},
      {w: "SUNNY",   m: "guessed"}
    ],
    neural: "CQ POTA DE K6XSS K", neural_conf: 0.9, neural_ok: true,
    neural_error: "", stretch: 2.4, prefer: "classic"
  }
};
class FakeWS {
  constructor(){
    this.readyState = 1;
    window.__ws = this;
    setTimeout(() => {
      this.onopen && this.onopen();
      this.onmessage && this.onmessage(
        {data: JSON.stringify({type: "hello", data: window.__state.hello})});
      this.onmessage && this.onmessage(
        {data: JSON.stringify({type: "decode", data: window.__state.decode})});
    }, 0);
  }
  send(raw){ try { window.__sent.push(JSON.parse(raw)); } catch(e){} }
  close(){}
}
window.WebSocket = FakeWS;
</script>
"""

# Runs after everything has settled: drives events and reports.
PROBE = r"""
<script>
// Wait for the page to actually have content rather than betting on a delay.
// A fixed timeout is a bet on machine speed, and a slow CI box loses it: the
// DOM gets dumped before the timer fires and a working page looks broken.
function whenReady(go, tries){
  // Wait for a decode to have actually rendered. #text exists in the static
  // page and window.__sent is created by the stub, so testing for those fires
  // before the scripted conversation is delivered and every transcript check
  // reads an empty pane.
  if (document.querySelector("#text .blk")) return go();
  if (tries <= 0) return go();
  setTimeout(() => whenReady(go, tries - 1), 50);
}
whenReady(() => {
  const r = {};
  const $ = s => document.querySelector(s);
  const click = el => el && el.dispatchEvent(
      new MouseEvent("click", {bubbles: true, cancelable: true}));
  const acts = () => window.__sent.map(m => m.action);

  r.no_script_error = !window.__err;
  r.error_text = window.__err || "";

  const text = $("#text");
  r.transcript_rendered = !!(text && text.textContent.trim());

  // The transcript must carry time anchors: an operator looking back at what
  // someone sent five minutes ago needs something to count from.
  r.has_timestamps = !!(text && text.querySelectorAll(".stamp").length >= 3);
  r.timestamp_looks_like_a_time =
      !!(text && /^\d\d:\d\d Z?$|^\d\d:\d\dZ$/.test(
          (text.querySelector(".stamp") || {}).textContent || ""));

  // Blocks, not one wall of text.
  r.blocks_are_separate = !!(text && text.querySelectorAll(".blk").length === 3);

  // Auto-log is the button an activator hits ten times in an hour.
  const auto = $("#auto-btn");
  r.auto_log_button_exists = !!auto;
  if (auto){
    const before = acts().filter(a => a === "autolog").length;
    click(auto);
    r.auto_log_sends_autolog =
        acts().filter(a => a === "autolog").length === before + 1;
  }

  // It must not be the same handler as the typed-entry button.
  const logbtn = $("#log-btn");
  r.typed_log_button_exists = !!logbtn;
  if (logbtn){
    const before = acts().filter(a => a === "autolog").length;
    click(logbtn);
    r.typed_log_is_a_different_action =
        acts().filter(a => a === "autolog").length === before;
  }

  const t = document.querySelector("a[href='/transcript']");
  r.transcript_link_exists = !!t;
  r.transcript_link_downloads = !!(t && t.hasAttribute("download"));

  // A cancelled word is struck and the run of dits collapses to one mark,
  // rather than printing eight Es at the operator.
  //
  // The pane renders only while it is open, so open it the way the operator
  // does and then let another decode arrive -- which is also the sequence
  // that revealed the pane stays blank until one does.
  const g = $("#guessed");
  r.guessed_pane_exists = !!g;
  if (g){
    const before_open = g.innerHTML;
    g.style.display = "block";
    r.pane_is_empty_until_a_decode_arrives = (before_open.trim() === "");
    window.__ws.onmessage(
      {data: JSON.stringify({type: "decode", data: window.__state.decode})});
    r.struck_word_rendered = !!g.querySelector(".g-struck");
    r.cancel_collapsed = g.querySelectorAll(".g-cancel").length === 1;
    r.cancel_is_not_a_wall_of_es =
        !/EEEEEE/.test(g.textContent || "");
  }

  r.stop_button_exists = !!$("#stop-btn");
  r.clear_button_exists = !!$("#clear-btn");
  document.title = "RESULT " + JSON.stringify(r);
}, 60);
</script>
"""


@unittest.skipUnless(CHROME, "no chromium installed")
class Interface(unittest.TestCase):
    results = None

    @classmethod
    def setUpClass(cls):
        page = PAGE.read_text()
        assert "<script>" in page, "the page has no script block"
        # STUB must land before the page's own script; PROBE after everything.
        page = page.replace("<script>", STUB + "<script>", 1)
        page = page.replace("</body>", PROBE + "</body>")
        with tempfile.NamedTemporaryFile("w", suffix=".html",
                                         delete=False) as fh:
            fh.write(page)
            path = fh.name
        try:
            out = subprocess.run(
                [CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                 "--virtual-time-budget=30000", "--dump-dom", f"file://{path}"],
                cwd=ROOT, capture_output=True, text=True, timeout=120)
        finally:
            Path(path).unlink(missing_ok=True)
        m = re.search(r"RESULT (\{.*?\})", out.stdout or "")
        if not m:
            # A browser that ran returns the rendered DOM whether or not the
            # probe fired; one that could not start returns nothing. Without
            # this split, a CI box with an unusable chromium reports a broken
            # interface.
            if "<html" not in (out.stdout or "").lower():
                raise unittest.SkipTest(
                    "chromium is installed but produced no DOM (exit %s): %s"
                    % (out.returncode, (out.stderr or "").strip()[-200:]))
            raise AssertionError(
                "the page rendered but the probe never ran -- a script error "
                "on load. stderr: " + (out.stderr or "").strip()[-300:])
        cls.results = json.loads(m.group(1))

    def check(self, key):
        self.assertIn(key, self.results, f"{key} did not run")
        self.assertTrue(self.results[key],
                        f"{key} (error: {self.results.get('error_text','')})")

    def test_the_page_loads_without_throwing(self):
        """A throw during load leaves the operator a blank screen."""
        self.check("no_script_error")

    def test_the_transcript_renders(self):
        self.check("transcript_rendered")

    def test_the_transcript_is_timestamped(self):
        """Someone copying by ear uses this pane to look back at a word that
        got away from him, and "five minutes ago" needs an anchor."""
        self.check("has_timestamps")
        self.check("timestamp_looks_like_a_time")

    def test_overs_are_separate_blocks(self):
        """Blocks break on silence so a net reads as separate overs rather
        than one unbroken wall."""
        self.check("blocks_are_separate")

    def test_auto_log_button_is_wired(self):
        self.check("auto_log_button_exists")
        self.check("auto_log_sends_autolog")

    def test_typed_log_is_a_separate_button(self):
        """Auto-log reads the contact off the air; the other logs what is in
        the boxes. Wiring both to one handler would be invisible until an
        activation went into the log wrong."""
        self.check("typed_log_button_exists")
        self.check("typed_log_is_a_different_action")

    def test_the_transcript_can_be_downloaded(self):
        self.check("transcript_link_exists")
        self.check("transcript_link_downloads")

    def test_a_cancelled_word_is_struck_not_printed(self):
        """The operator sends a run of dits and starts again. Printing them
        restates the problem the feature exists to solve."""
        self.check("guessed_pane_exists")
        self.check("struck_word_rendered")
        self.check("cancel_collapsed")
        self.check("cancel_is_not_a_wall_of_es")

    def test_the_stop_and_clear_buttons_exist(self):
        self.check("stop_button_exists")
        self.check("clear_button_exists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
