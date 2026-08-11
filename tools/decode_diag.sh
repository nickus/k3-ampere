#!/bin/bash
# PP=4 boots and its prefill taps are bit-identical to PP=1, but the first
# DECODE step deadlocks: the engine dies waiting on a worker response. Same
# technique that localised the warmup hang — serve, fire one request, and when
# it does not come back, have every worker dump itself via SIGUSR1.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
PPN=${PPN:-4}
DEADLINE=${DEADLINE:-120}          # seconds to wait for the request itself
OUT=/workspace/k3/decode_diag
rm -rf "$OUT"; mkdir -p "$OUT"

export PYTHONPATH=/workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DSPARK_PROBE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export SKIP_KERNEL_WARMUP=1        # warmup is a separate, already-diagnosed hang

cat > sitecustomize.py <<'EOF'
try:
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception as e:
    print("[FAULTHANDLER] not installed:", e, flush=True)
try:
    import dspark_pp_probe  # noqa: F401
except Exception as e:
    print("[DSPARK_PROBE] not installed:", e, flush=True)
EOF

SPEC='{"model":"/workspace/k3/k3-dspark-draft","method":"dspark","num_speculative_tokens":3}'
LOG="$OUT/server.log"
pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 5

nohup $PY -m vllm.entrypoints.openai.api_server \
  --model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code \
  --load-format dummy --pipeline-parallel-size "$PPN" --tensor-parallel-size 1 \
  --speculative-config "$SPEC" \
  --enable-prefix-caching --block-size 512 \
  --max-model-len 4096 --gpu-memory-utilization 0.82 --enforce-eager \
  --port 18000 > "$LOG" 2>&1 &

for i in $(seq 1 60); do
  sleep 10
  curl -s --max-time 5 http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q '"id"' && break
done
if ! curl -s --max-time 5 http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q '"id"'; then
  echo "SERVER_NEVER_CAME_UP" | tee "$OUT/state.txt"; tail -30 "$LOG" > "$OUT/log_tail.txt"; exit 1
fi
echo "served at PP=$PPN, firing one request"

MARK=$(wc -l < "$LOG")
curl -s --max-time "$DEADLINE" http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","prompt":"The quick brown fox jumps over the lazy dog and then","max_tokens":24,"temperature":0}' \
  > "$OUT/response.json" 2>&1
RC=$?

if [ $RC -eq 0 ] && grep -q '"text"' "$OUT/response.json" 2>/dev/null; then
  echo "STATE=served_ok" | tee "$OUT/state.txt"
else
  echo "STATE=decode_stuck (curl rc=$RC)" | tee "$OUT/state.txt"
  for pid in $(pgrep -f "[V]LLM::"); do
    echo "=== signalling pid=$pid $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-60)" >> "$OUT/stacks.txt"
    kill -USR1 "$pid" 2>/dev/null
  done
  sleep 15
  tail -n +$((MARK + 1)) "$LOG" >> "$OUT/stacks.txt"
fi
grep -E "DSPARK_PROBE|ERROR|Traceback" "$LOG" | tail -20 > "$OUT/markers.txt"
echo "--- collected in $OUT"
