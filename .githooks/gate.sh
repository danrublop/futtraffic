#!/bin/sh
# A gate that was KILLED did not fail. It did not run.
#
# 2026-09-19, twice in one night: `verify_claims --all` was SIGTERMed because
# the push was running as a long background task, the hook saw a non-zero exit
# and printed "push blocked: a claim drifted". Nothing had drifted. The message
# sent a person hunting for a number that had not moved, which is the most
# expensive thing a gate can do - it spends the trust that makes anyone run it.
#
# A shell reports a signal death as 128 + N. That is the whole distinction, and
# it is one comparison. The push is still refused either way: a claim that did
# not finish did not verify. Only the sentence changes, and the sentence is the
# entire product of a gate that blocks.
#
#   run_gate "<label>" cmd args...      # 0 = pass, non-zero = refuse
#
# The caller prints the "what to do about it" text on a non-zero return; this
# prints the "what actually happened" line, which is the part that was wrong.
#
#   sh .githooks/gate.sh --selfcheck    # both branches, on real processes

run_gate() {
  label=$1
  shift
  "$@"
  st=$?
  [ "$st" -eq 0 ] && return 0
  if [ "$st" -ge 128 ]; then
    echo ""
    echo "  $label was KILLED by signal $((st - 128)) - it did not finish."
    echo "  NOTHING WAS MEASURED, so nothing drifted. A push run as a background"
    echo "  task and then reaped lands here, and so does an OOM kill."
    echo "  Re-run it in the FOREGROUND before you go looking for a number."
    return 2
  fi
  echo ""
  echo "  $label ran to completion and refused the push."
  return 1
}

if [ "$1" = "--selfcheck" ]; then
  out=$(run_gate "a killed gate" sh -c 'kill -TERM $$' 2>&1); rc=$?
  [ "$rc" = 2 ] || { echo "FAIL: a SIGTERMed gate returned $rc, wanted 2"; exit 1; }
  case "$out" in *KILLED*) ;; *) echo "FAIL: no KILLED in: $out"; exit 1 ;; esac
  # the exact sentence that cost two cycles must not be reachable from here
  case "$out" in *"a claim drifted"*) echo "FAIL: a killed gate blamed drift"; exit 1 ;; esac

  out=$(run_gate "a failing gate" sh -c 'exit 1' 2>&1); rc=$?
  [ "$rc" = 1 ] || { echo "FAIL: a failing gate returned $rc, wanted 1"; exit 1; }
  case "$out" in *KILLED*) echo "FAIL: a plain failure was read as killed"; exit 1 ;; esac

  run_gate "a passing gate" true >/dev/null || { echo "FAIL: true did not pass"; exit 1; }
  echo "gate.sh self-check OK (killed -> 2 and says so, failed -> 1, passed -> 0)"
  exit 0
fi
