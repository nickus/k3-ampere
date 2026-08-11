#!/bin/bash
# Diagnose the DSpark-under-PP warmup hang by reading the stack of the process
# that is actually stuck, instead of adding another round of print probes.
#
# Previous probes localised it to: rank 0 blocked in pp_handler.receive inside
# sample_tokens, last rank inside self.sample. That says "the last rank never
# finishes sampling" but not why. py-spy answers the why in one run.
#
# Usage:  PPN=4 bash warmup_diag.sh
# Output: warmup_diag/  (stacks, log tail, nvidia-smi)
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
PPN=${PPN:-2}
OUT=/workspace/k3/warmup_diag
rm -rf "$OUT"; mkdir -p "$OUT"

export PYTHONPATH=/workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DSPARK_PROBE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# FlashInfer's sampler is JIT-built on first use, on the LAST PP rank only,
# while every other rank sits blocked in pp_handler.receive. On an image
# without nvcc/ninja the build cannot even start (FileNotFoundError: 'ninja')
# and the whole engine dies inside warmup; with a toolchain present it is a
# multi-minute compile that is indistinguishable from a deadlock. The torch
# sampler is correct and needs no toolchain.
export VLLM_USE_FLASHINFER_SAMPLER=0
# NOTE: SKIP_KERNEL_WARMUP is deliberately NOT set — the hang is the subject.
unset SKIP_KERNEL_WARMUP

cat > sitecustomize.py <<'EOF'
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

# Wait for either: server up (no hang — warmup fixed), or the log going quiet
# after warmup was entered (the hang), or an outright crash.
STALL=0; PREV=0; STATE=unknown
for i in $(seq 1 120); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q '"id"'; then
    STATE=served; break
  fi
  if grep -qiE "Traceback|CUDA error|illegal memory|NotImplementedError" "$LOG"; then
    STATE=crashed; break
  fi
  SZ=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$SZ" = "$PREV" ]; then STALL=$((STALL+1)); else STALL=0; PREV=$SZ; fi
  # 90s of no new output and still not serving => hung. Deliberately NOT keyed on
  # the probe's "ENTER warmup_kernels" marker: the point of this run is to work
  # without instrumentation, and a stack tells us the phase anyway.
  if [ "$STALL" -ge 9 ]; then STATE=hung; break; fi
done
echo "STATE=$STATE (after $((i*10))s)" | tee "$OUT/state.txt"

if [ "$STATE" = "hung" ]; then
  command -v py-spy >/dev/null || $PY -m pip install -q py-spy 2>/dev/null
  PYSPY=$(command -v py-spy || echo /venv/nm/bin/py-spy)
  for pid in $(pgrep -f "[V]LLM::" ) $(pgrep -f "[E]ngineCore"); do
    NAME=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-90)
    echo "=== pid=$pid $NAME" >> "$OUT/stacks.txt"
    timeout 90 "$PYSPY" dump --pid "$pid" --locals >> "$OUT/stacks.txt" 2>&1
    echo >> "$OUT/stacks.txt"
  done
  nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  # Are any CUDA kernels actually resident, or is everyone idle in a wait?
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv \
    >> "$OUT/nvidia-smi.txt" 2>&1
fi

grep -E "\[WM\]|ENTER warmup|Traceback|Error" "$LOG" | tail -40 > "$OUT/markers.txt"
tail -60 "$LOG" > "$OUT/log_tail.txt"
echo "--- collected in $OUT"
