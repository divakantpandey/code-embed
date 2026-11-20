#!/bin/bash
# On-instance completion guard, independent of any local watcher.
# Destroys this instance only after verify_uploads.py proves model + results are on the Hub.
# Failure or a vanished driver stops (not destroys) it, so disk, logs and checkpoints survive for inspection.
set -uo pipefail
W=/workspace; LOG=$W/guard.log; PY=$W/code/.venv/bin/python
while IFS= read -r -d "" kv; do
  case "$kv" in HF_TOKEN=*|CONTAINER_API_KEY=*|CONTAINER_ID=*) export "$kv";; esac
done < /proc/1/environ
export HF_HOME=$W/.cache/huggingface
VAST="$HOME/.local/bin/vastai"
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
push_logs() {
  cp -f $W/*.log $W/out/ 2>/dev/null
  $PY -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='$W/out', repo_id='divakantpandey/code-embed-v3-src', path_in_repo='v2-zeroshot/run',
  allow_patterns=['*.log', 'progress.json', 'stages.json', 'throughput.json'], commit_message='v2-zeroshot: guard $1')" >> $LOG 2>&1 \
  || log "log upload failed"
}
halt() {  # $1 = stop|destroy
  push_logs "$1"
  for i in 1 2 3; do
    log "vastai $1 instance $CONTAINER_ID (attempt $i)"
    # destroy prompts without -y and then exits 0 on "Aborted"; stop has no -y flag at all.
    if [ "$1" = destroy ]; then
      "$VAST" destroy instance "$CONTAINER_ID" -y --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
    else
      "$VAST" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> $LOG 2>&1
    fi
    sleep 60
    # Still running here means the call did not take effect (a destroy or stop kills this container), so read the
    # state instead of trusting the exit code.
    st=$("$VAST" show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --raw 2>/dev/null \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('intended_status'))" 2>/dev/null)
    log "intended_status after attempt $i: ${st:-unreadable}"
    [ "$1" = stop ] && [ "$st" = stopped ] && exit 0
  done
  log "vastai $1 did not take effect after 3 attempts"; exit 1
}

log "armed for instance $CONTAINER_ID"
tries=0; gone=0
while true; do
  note=$(cat $W/out/progress.json 2>/dev/null)
  case "$note" in
    *'"all done"'*)
      if $PY $W/verify_uploads.py >> $LOG 2>&1; then log "outputs verified"; halt destroy; fi
      tries=$((tries + 1)); log "verification failed ($tries/3)"
      [ $tries -ge 3 ] && { log "cannot verify uploads, stopping instead of destroying"; halt stop; }
      sleep 600; continue ;;
    *'"FAILED'*)
      log "run failed: $note"; halt stop ;;
  esac
  if pgrep -f "bash $W/run.sh" >/dev/null; then gone=0
  else gone=$((gone + 1)); log "driver not running ($gone/5)"
    [ $gone -ge 5 ] && { log "driver gone without a final status"; halt stop; }
  fi
  sleep 120
done
