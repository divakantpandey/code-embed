#!/bin/bash
# Local status logger + fallback for the v2-zeroshot run. The on-instance guard.sh is the primary
# destroyer (it verifies Hub uploads first); this only destroys if the instance is still up 30 min
# after "all done", i.e. the guard died. Never destroys for cost: the run must finish.
ID=61413883; FLOOR=1.50; done_at=""
ist() { TZ=Asia/Kolkata date +"%d %b %H:%M IST"; }
while true; do
  credit=$(vastai show user --raw 2>/dev/null | python3 -c "import json,sys; print(round(json.load(sys.stdin)['credit'],2))" 2>/dev/null)
  note=$(curl -fsSL -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/divakantpandey/code-embed-v3-src/resolve/main/v2-zeroshot/run/progress.json 2>/dev/null)
  state=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys; print(next((i['actual_status'] for i in json.load(sys.stdin) if i['id']==$ID), 'gone'))" 2>/dev/null)
  echo "$(ist) credit=$credit state=$state $note"
  [ "$state" = "gone" ] && { echo "INSTANCE GONE (guard destroyed it, or it was removed)"; exit 0; }
  if [ -n "$credit" ] && python3 -c "import sys; sys.exit(0 if float('$credit') < $FLOOR else 1)"; then
    echo "LOW CREDIT: $credit, top-up needed"; exit 0; fi
  case "$note" in
    *"all done"*)
      [ -z "$done_at" ] && done_at=$(date +%s)
      if [ $(( $(date +%s) - done_at )) -ge 1800 ] && [ "$state" = "running" ]; then
        vastai destroy instance $ID -y; echo "FALLBACK DESTROY: guard did not act within 30 min"; exit 0; fi ;;
    *FAILED*) echo "RUN FAILED: $note (guard stops the instance, disk kept)"; exit 0 ;;
  esac
  # "offline" means Vast lost contact with the host agent, often briefly while the job keeps running; only warn.
  case "$state" in exited|stopped|unknown) echo "INSTANCE $state"; exit 0 ;; offline) echo "WARN: Vast reports host offline" ;; esac
  sleep 300
done
