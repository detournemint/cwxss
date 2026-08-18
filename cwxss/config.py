"""Station settings and the messages the operator sends.

Kept in a plain JSON file so it can be edited with any text editor in a field,
on a laptop with no network, without going near the code.
"""
import json
import os
from pathlib import Path

PATH = Path(os.environ.get("CWXSS_CONFIG",
                           Path.home() / ".config" / "cwxss.json"))

# A POTA activation is a small number of sentences repeated all afternoon, so
# the defaults are that exchange, in order. {call} is our callsign, {his} the
# station being worked, {rst} the report, {state} our location.
DEFAULTS = {
    "call": "N0CALL",
    "state": "CA",
    "park": "",
    "wpm": 20,
    "rst": "5NN",
    # Two sets, because the two ways of operating want different words. A park
    # activation is a short exchange repeated all afternoon; a conversation is
    # not, and having to skip past POTA macros to find "name is" is friction at
    # exactly the wrong moment.
    "active": "pota",
    "sets": {
        "pota": [
            {"key": "F1", "label": "CQ POTA",  "text": "CQ POTA DE {call} {call} K"},
            {"key": "F2", "label": "his+rprt", "text": "{his} {rst} {rst} {state}"},
            {"key": "F3", "label": "roger TU", "text": "RR TU {call} K"},
            {"key": "F4", "label": "my call",  "text": "{call}"},
            {"key": "F5", "label": "again?",   "text": "AGN AGN"},
            {"key": "F6", "label": "his call?", "text": "{his}?"},
            {"key": "F7", "label": "73",       "text": "TU 73 GL DE {call}"},
            {"key": "F8", "label": "QRZ",      "text": "QRZ DE {call} K"},
            {"key": "F9", "label": "park",     "text": "PARK IS {park} {park}"},
            {"key": "F10", "label": "QRL?",    "text": "QRL?"},
        ],
        "general": [
            {"key": "F1", "label": "CQ",       "text": "CQ CQ DE {call} {call} K"},
            {"key": "F2", "label": "answer",   "text": "{his} DE {call} GE"},
            {"key": "F3", "label": "rprt",     "text": "UR {rst} {rst} IN {state}"},
            {"key": "F4", "label": "name+qth", "text": "NAME IS {name} QTH {state}"},
            {"key": "F5", "label": "rig",      "text": "RIG IS {rig} PWR {watts}W"},
            {"key": "F6", "label": "wx",       "text": "WX HR IS {wx}"},
            {"key": "F7", "label": "how copy", "text": "HW CPY? {his} DE {call} K"},
            {"key": "F8", "label": "73",       "text": "TNX FER QSO 73 ES GL DE {call}"},
            {"key": "F9", "label": "again?",   "text": "PSE AGN"},
            {"key": "F10", "label": "QRL?",    "text": "QRL?"},
        ],
    },
    "name": "",
    "rig": "FT991",
    "watts": "50",
    "wx": "FB",
}


def load():
    cfg = json.loads(json.dumps(DEFAULTS))       # a copy, not a shared object
    try:
        cfg.update(json.loads(PATH.read_text()))
    except (OSError, ValueError):
        pass
    # An older config kept one flat list. Keep it working rather than silently
    # replacing the operator's own macros with the defaults.
    if isinstance(cfg.get("messages"), list) and cfg["messages"]:
        cfg.setdefault("sets", {})["pota"] = cfg.pop("messages")
    cfg.setdefault("sets", DEFAULTS["sets"])
    if cfg.get("active") not in cfg["sets"]:
        cfg["active"] = next(iter(cfg["sets"]))
    return cfg


def messages(cfg):
    """The macro set currently selected."""
    return cfg.get("sets", {}).get(cfg.get("active", "pota"), [])


def save(cfg):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cfg, indent=2))
    return PATH


def expand(text, cfg, his=""):
    """Fill in the placeholders. An empty field leaves no stray token behind."""
    out = (text or "")
    for token, value in (("{call}", cfg.get("call", "")),
                         ("{his}", his or ""),
                         ("{rst}", cfg.get("rst", "5NN")),
                         ("{state}", cfg.get("state", "")),
                         ("{park}", cfg.get("park", "")),
                         ("{name}", cfg.get("name", "")),
                         ("{rig}", cfg.get("rig", "")),
                         ("{watts}", cfg.get("watts", "")),
                         ("{wx}", cfg.get("wx", ""))):
        out = out.replace(token, str(value))
    return " ".join(out.upper().split())
