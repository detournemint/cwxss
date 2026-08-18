#!/bin/bash
# Harvest during a known activity window, then put the station back.
#
# Blind sweeping and idle afternoons both produce almost nothing -- six
# reachable stations on a Tuesday afternoon, of which maybe two are copyable.
# Organised events produce an order of magnitude more, in a known hour, with
# every station calling CQ and the Reverse Beacon Network naming each one.
#
# Called by systemd timers. Takes the audio device from cwxss for the duration
# and gives it back afterwards, whatever happens.
set -u
MINUTES="${1:-60}"
LOG=/tmp/cwxss-harvest-$(date -u +%Y%m%d-%H%M).log

restore() { systemctl --user start cwxss 2>/dev/null; }
trap restore EXIT INT TERM

{
  echo "=== harvest window: ${MINUTES} min from $(date -u +%H:%MZ) ==="
  systemctl --user stop cwxss
  sleep 2
  python3 "$HOME/cwxss/harvest.py" --rbn 45 --minutes "$MINUTES" \
      --record 40 --out "$HOME/cwdata"
  echo "=== done, $(ls -1 "$HOME"/cwdata/*.wav 2>/dev/null | wc -l) total captures ==="
} >>"$LOG" 2>&1
