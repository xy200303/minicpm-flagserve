#!/usr/bin/env bash
# Watch vllm serve log; on a zero-generation stall, dump py-spy stacks.
LOG=$1
OUT=/workspace/stall_dumps
mkdir -p "$OUT"
LAST=0
tail -n0 -F "$LOG" | while read -r line; do
  case "$line" in
    *"generation throughput: 0.0"*)
      now=$(date +%s)
      if [ $((now - LAST)) -lt 30 ]; then continue; fi
      LAST=$now
      ts=$(date +%H%M%S)
      echo "STALL at $ts" >> "$OUT/events.txt"
      for pid in $(ps aux | grep -E 'EngineCor|APIServer|vllm serve' | grep -v grep | awk '{print $2}'); do
        py-spy dump --pid "$pid" > "$OUT/dump_${ts}_${pid}.txt" 2>&1
      done
      mx-smi > "$OUT/mxsmi_${ts}.txt" 2>&1
      ;;
  esac
done
