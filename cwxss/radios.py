"""Per-radio profiles.

Radios differ in the ways that matter here and agree on very little. What baud
rate, which of two serial ports is CAT, how the computer keys it, and whether
hamlib's idea of the rig can be trusted -- all of it is per-model, and getting
any of it wrong produces silence rather than an error.

So the knowledge lives in one table rather than being rediscovered by each
operator. Everything here is overridable: a profile is a good starting point,
not a claim about your particular setup.
"""

# keying:
#   "line"   toggle a serial control line -- what flrig and fldigi do, works on
#            anything with a keying input, and the timing is ours to generate
#   "hamlib" hand text to the rig's own keyer via hamlib send_morse
#   "none"   no keyboard CW
PROFILES = {
    "ft991a": {
        "name": "Yaesu FT-991 / FT-991A",
        "hamlib_model": 1035,
        "baud": 38400,
        "keying": "line",
        "key_signal": "dtr",
        # Two ports on one chip. The first is CAT, the second is for keying,
        # and swapping them is the commonest reason keyboard CW does nothing.
        "cat_interface": 0,
        "key_interface": 1,
        "audio_hint": "USB Audio CODEC",
        "cw_pitch_default": 700,
        "setup_notes": [
            "Menu 060 PC KEYING must be set to DTR. It is off from the factory,"
            " and with it off the software sends perfectly and no RF comes out.",
            "hamlib send_morse does not work on this rig: it uses the CAT KY"
            " command, which only replays stored keyer memories, and hamlib's"
            " check of whether the keyer is free asks a question the rig does"
            " not implement. Use the key line.",
        ],
        "menu_commands": {"pc_keying_dtr": "EX0603;", "pc_keying_off": "EX0600;"},
    },
    "x6100": {
        "name": "Xiegu X6100",
        "hamlib_model": 3087,
        # The backend advertises 300..19200. 19200 is the usual setting on the
        # radio; if CAT does not answer, this is the first thing to check.
        "baud": 19200,
        # The X6100 presents a single CAT port and hamlib reports no morse
        # sending for it, so keying is by control line into the KEY jack, or by
        # PTT plus a sidetone if no keying interface is present.
        "keying": "line",
        "key_signal": "dtr",
        "cat_interface": 0,
        "key_interface": 0,          # same port; DTR is free while CAT uses TX/RX
        "audio_hint": "USB Audio",
        "cw_pitch_default": 700,
        "setup_notes": [
            "The X6100 exposes one USB serial port for CAT. Its DTR line is"
            " unused by CAT and can key the radio through a simple interface"
            " into the KEY jack.",
            "hamlib does not report morse sending for this model, so keyboard"
            " CW goes through the key line rather than the radio's keyer.",
            "Check the radio's CW settings: full break-in makes keyboard"
            " sending far more usable, and the keyer must be set to accept an"
            " external key rather than paddles only.",
        ],
        "menu_commands": {},
    },
    "g90": {
        "name": "Xiegu G90",
        "hamlib_model": 3088,
        "baud": 19200,
        "keying": "line",
        "key_signal": "dtr",
        "cat_interface": 0,
        "key_interface": 0,
        "audio_hint": "USB Audio",
        "cw_pitch_default": 700,
        "setup_notes": ["The G90 has no built-in sound card on some revisions;"
                        " audio may come from a separate interface."],
        "menu_commands": {},
    },
    "generic": {
        "name": "Any rig hamlib supports",
        "hamlib_model": 2,           # NET rigctl
        "baud": 38400,
        "keying": "line",
        "key_signal": "dtr",
        "cat_interface": 0,
        "key_interface": 1,
        "audio_hint": "USB Audio",
        "cw_pitch_default": 700,
        "setup_notes": ["If keyboard CW does nothing, the radio almost"
                        " certainly needs telling to accept keying from the"
                        " computer. Look for a PC KEYING or similar menu item."],
        "menu_commands": {},
    },
}


def get(key):
    return PROFILES.get((key or "").lower().replace("-", "").replace(" ", ""),
                        PROFILES["generic"])


def by_hamlib_model(model):
    for k, p in PROFILES.items():
        if p["hamlib_model"] == int(model or 0):
            return k, p
    return "generic", PROFILES["generic"]


def choices():
    return [(k, p["name"], p["hamlib_model"]) for k, p in PROFILES.items()]
