#!/usr/bin/env bash
#
# cwxss setup.
#
# Installs what is needed, finds your radio, works out which serial port is
# which, tests that everything responds, and writes a config. It asks before
# every change and is safe to re-run.
#
#   ./setup.sh              set up on this machine
#   ./setup.sh --service    also install a systemd user service
#   ./setup.sh --check      diagnose an existing install, change nothing
#
set -uo pipefail

SERVICE=0; CHECK=0; YES=0
for a in "$@"; do case "$a" in
  --service) SERVICE=1 ;; --check) CHECK=1 ;; -y|--yes) YES=1 ;;
  -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
  *) echo "unknown option: $a"; exit 2 ;;
esac; done

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HOME/.config/cwxss.json"
ok(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
no(){ printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m!\033[0m %s\n' "$*"; }
step(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
ask(){ [ "$YES" = 1 ] && { echo "$2"; return; }
       local r; read -r -p "  $1 [$2]: " r </dev/tty; echo "${r:-$2}"; }
yn(){ [ "$YES" = 1 ] && return 0
      local r; read -r -p "  $1 [y/N]: " r </dev/tty; [[ "$r" =~ ^[Yy] ]]; }

printf '\033[1mcwxss setup\033[0m\n'

# ------------------------------------------------------------ dependencies --
step "1. Dependencies"
if   command -v apt-get >/dev/null; then PM=apt
elif command -v dnf     >/dev/null; then PM=dnf
elif command -v pacman  >/dev/null; then PM=pacman
else PM=""; fi
case "$PM" in
  apt)    PKGS=(python3 python3-numpy python3-aiohttp python3-serial alsa-utils libhamlib-utils) ;;
  dnf)    PKGS=(python3 python3-numpy python3-aiohttp python3-pyserial alsa-utils hamlib) ;;
  pacman) PKGS=(python python-numpy python-aiohttp python-pyserial alsa-utils hamlib) ;;
esac
if [ "$CHECK" = 0 ] && [ -n "$PM" ]; then
  echo "  will install: ${PKGS[*]}"
  if yn "Install with $PM (needs sudo)?"; then
    case "$PM" in
      apt) sudo apt-get update -qq && sudo apt-get install -y "${PKGS[@]}" ;;
      dnf) sudo dnf install -y "${PKGS[@]}" ;;
      pacman) sudo pacman -S --needed --noconfirm "${PKGS[@]}" ;;
    esac
  fi
fi
for m in numpy aiohttp serial; do
  python3 -c "import $m" 2>/dev/null && ok "python $m" || no "python $m missing"
done
command -v arecord >/dev/null && ok "arecord" || no "arecord missing (alsa-utils)"
command -v rigctld >/dev/null && ok "rigctld" || warn "rigctld missing — no CAT, decode still works"

# ------------------------------------------------------------------ audio --
step "2. Audio: which device is the radio?"
mapfile -t CARDS < <(arecord -l 2>/dev/null | grep ^card)
if [ "${#CARDS[@]}" -eq 0 ]; then
  no "no capture devices at all — is the radio plugged in and switched on?"
else
  i=0; declare -a DEVS=()
  for line in "${CARDS[@]}"; do
    c=$(sed 's/card \([0-9]*\).*/\1/' <<<"$line")
    d=$(sed 's/.*device \([0-9]*\).*/\1/' <<<"$line")
    name=$(sed 's/.*\[\(.*\)\].*device.*/\1/' <<<"$line")
    DEVS+=("plughw:$c,$d")
    # A radio's USB audio is a generic codec chip; the built-in mic and a webcam
    # are the two things people record by mistake.
    hint=""
    grep -qiE "codec|usb audio" <<<"$name" && hint="  <- looks like a radio"
    grep -qiE "webcam|kiyo|camera|hd audio|pch|generic analog" <<<"$name" && hint="  (not a radio)"
    printf "    [%d] %-14s %s%s\n" "$i" "plughw:$c,$d" "$name" "$hint"
    i=$((i+1))
  done
  DEF=0
  for n in "${!DEVS[@]}"; do
    grep -qiE "codec|usb audio" <<<"${CARDS[$n]}" && { DEF=$n; break; }
  done
  SEL=$(ask "Which device is the radio's receive audio?" "$DEF")
  AUDIO="${DEVS[$SEL]:-${DEVS[0]}}"
  ok "audio: $AUDIO"
fi

# ----------------------------------------------------------------- serial --
step "3. Serial: CAT and the keying line"
mapfile -t PORTS < <(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null)
CAT=""; KEYLINE=""
if [ "${#PORTS[@]}" -eq 0 ]; then
  warn "no serial ports — CAT and keyboard keying will not work"
else
  for p in "${PORTS[@]}"; do
    info=$(udevadm info -q property -n "$p" 2>/dev/null)
    model=$(sed -n 's/^ID_MODEL=//p' <<<"$info")
    iface=$(sed -n 's/^ID_USB_INTERFACE_NUM=//p' <<<"$info")
    printf "    %-14s %s (interface %s)\n" "$p" "${model:-unknown}" "${iface:-?}"
  done
  # Many radios present two ports on one chip: the first is CAT, the second is
  # for keying. Getting these the wrong way round is the single most common
  # reason keyboard CW does nothing at all.
  CAT=$(ask "Which port is CAT control?" "${PORTS[0]}")
  KEYLINE=$(ask "Which port is the CW keying line? (blank for none)" \
                "${PORTS[1]:-}")
  [ -n "$CAT" ] && ok "CAT: $CAT"
  [ -n "$KEYLINE" ] && ok "keying: $KEYLINE"
fi
if ! id -nG "$USER" | grep -qwE "dialout|uucp"; then
  warn "you are not in the 'dialout' group; opening serial ports will fail"
  yn "Add $USER to dialout?" && sudo usermod -aG dialout "$USER" \
     && warn "log out and back in for that to take effect"
fi

# -------------------------------------------------------------------- rig --
step "4. Radio"
MODEL=$(ask "hamlib rig model (rigctl --list to search; 1035 = Yaesu FT-991/A)" "1035")
BAUD=$(ask "CAT baud rate" "38400")
if [ -n "$CAT" ] && command -v rigctl >/dev/null && [ "$CHECK" = 0 ]; then
  if yn "Test CAT now?"; then
    if out=$(timeout 8 rigctl -m "$MODEL" -r "$CAT" -s "$BAUD" f 2>&1); then
      ok "radio replied: $out Hz"
      # Yaesu radios will not key from a control line until this is enabled,
      # and the setting is off from the factory. It cost an evening to find.
      if [[ "$MODEL" == "1035" ]] && [ -n "$KEYLINE" ]; then
        cur=$(timeout 8 rigctl -m "$MODEL" -r "$CAT" -s "$BAUD" w "EX060;" 2>/dev/null)
        echo "    menu 060 PC KEYING currently: ${cur:-unknown}"
        if yn "Set menu 060 PC KEYING to DTR (required for keyboard CW)?"; then
          timeout 8 rigctl -m "$MODEL" -r "$CAT" -s "$BAUD" w "EX0603;" >/dev/null 2>&1
          ok "PC KEYING set to DTR"
        fi
      fi
    else
      no "no reply: $out"
      warn "check the port, baud rate, model, and that CAT is enabled on the radio"
    fi
  fi
fi

# ----------------------------------------------------------------- config --
step "5. Station"
CALL=$(ask "Callsign" "$(python3 -c "
import json,sys
try: print(json.load(open('$CONFIG')).get('call','N0CALL'))
except Exception: print('N0CALL')" 2>/dev/null)")
STATE=$(ask "State or province (sent in the exchange)" "CA")
NAME=$(ask "Your name (for ragchew macros)" "")
WPM=$(ask "Default keyer speed in wpm" "20")
if [ "$CHECK" = 0 ]; then
  mkdir -p "$(dirname "$CONFIG")"
  python3 - "$CONFIG" "$CALL" "$STATE" "$NAME" "$WPM" <<'PY'
import json, sys
path, call, state, name, wpm = sys.argv[1:6]
try:
    cfg = json.load(open(path))
except Exception:
    cfg = {}
cfg.update({"call": call.upper(), "state": state.upper(),
            "name": name.upper(), "wpm": int(wpm)})
json.dump(cfg, open(path, "w"), indent=2)
PY
  ok "wrote $CONFIG"
fi

# ---------------------------------------------------------------- service --
CMD="python3 $SRC/cwxss/server.py --device ${AUDIO:-plughw:1,0} --port 8074 --bind 0.0.0.0"
[ -n "$CAT" ] && CMD="$CMD --rig 127.0.0.1:4532"
[ -n "$KEYLINE" ] && CMD="$CMD --keyline $KEYLINE --keyline-signal dtr"

if [ "$SERVICE" = 1 ] && [ "$CHECK" = 0 ]; then
  step "6. Service"
  U="$HOME/.config/systemd/user"; mkdir -p "$U"
  if [ -n "$CAT" ]; then
    cat > "$U/cwxss-rigctld.service" <<EOF
[Unit]
Description=hamlib rigctld for cwxss
[Service]
ExecStart=/usr/bin/rigctld -m $MODEL -r $CAT -s $BAUD -T 127.0.0.1 -t 4532
Restart=always
RestartSec=3
[Install]
WantedBy=default.target
EOF
    ok "cwxss-rigctld.service"
  fi
  cat > "$U/cwxss.service" <<EOF
[Unit]
Description=cwxss - CW decoder and keyboard keyer
After=cwxss-rigctld.service
Wants=cwxss-rigctld.service
[Service]
WorkingDirectory=$SRC
ExecStart=/usr/bin/$CMD
Restart=always
RestartSec=3
[Install]
WantedBy=default.target
EOF
  ok "cwxss.service"
  systemctl --user daemon-reload
  yn "Enable and start now?" && {
    [ -n "$CAT" ] && systemctl --user enable --now cwxss-rigctld >/dev/null 2>&1
    systemctl --user enable --now cwxss >/dev/null 2>&1
    sleep 3
    systemctl --user is-active --quiet cwxss && ok "running" \
      || no "failed — journalctl --user -u cwxss -n 30"
  }
fi

step "Done"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "  Open  http://${IP:-localhost}:8074"
echo
echo "  To run it by hand:"
echo "    $CMD"
echo
echo "  Try it with no radio at all:"
echo "    python3 $SRC/cwxss/server.py --demo"
if [ -n "$KEYLINE" ]; then
  echo
  echo "  Keyboard CW needs the radio configured to accept keying from the"
  echo "  computer. On a Yaesu that is menu 060 PC KEYING = DTR. Without it"
  echo "  the software will appear to send perfectly and no RF will come out."
fi
