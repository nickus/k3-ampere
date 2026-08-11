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
# Dump every thread's stack on SIGUSR1. py-spy cannot be used here: vast.ai
# containers are not given CAP_SYS_PTRACE, so it dies with "Failed to copy
# Py_Version symbol: Permission denied". faulthandler needs no privileges
# because the process dumps itself; output goes to stderr, i.e. the server log.
try:
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception as e:  # never let instrumentation break the server
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

# Wait for either: server up (no hang — warmup fixed), or the log going quiet
# after warmup was entered (the hang), or an outright crash.
# Must fire well before NCCL's 600s watchdog, which SIGABRTs the workers and
# takes the stacks with it. Warmup itself is minutes, so 300s is a safe middle.
DEADLINE=${DEADLINE:-300}
STALL=0; PREV=0; STATE=unknown
for i in $(seq 1 120); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q '"id"'; then
    STATE=served; break
  fi
  if grep -qiE "Traceback|CUDA error|illegal memory|NotImplementedError" "$LOG"; then
    STATE=crashed; break
  fi
  # Deadline, not stall detection. A size-based stall counter looks obvious and
  # does not work here: vLLM prints "No available shared memory broadcast block
  # found in 60 seconds" on a timer, so the log keeps growing while nothing
  # progresses, the counter resets every minute, and the run sails past the
  # point worth sampling straight into the 600s NCCL watchdog — which kills the
  # processes before any stack can be taken. Whatever the engine is doing, if it
  # has not served in DEADLINE seconds it is stuck, and the stack says where.
  if [ $((i * 10)) -ge "$DEADLINE" ]; then STATE=hung; break; fi
done
echo "STATE=$STATE (after $((i*10))s)" | tee "$OUT/state.txt"

if [ "$STATE" = "hung" ]; then
  # Ask each worker to dump itself. The stacks land in its stderr, which is this
  # same log, so record where to start reading from.
  MARK=$(wc -l < "$LOG")
  for pid in $(pgrep -f "[V]LLM::"); do
    echo "=== signalling pid=$pid $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-60)" \
      >> "$OUT/stacks.txt"
    kill -USR1 "$pid" 2>/dev/null
  done
  sleep 15
  tail -n +$((MARK + 1)) "$LOG" >> "$OUT/stacks.txt"
  nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  # Are any CUDA kernels actually resident, or is everyone idle in a wait?
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv \
    >> "$OUT/nvidia-smi.txt" 2>&1
fi

grep -E "\[WM\]|ENTER warmup|Traceback|Error" "$LOG" | tail -40 > "$OUT/markers.txt"
tail -60 "$LOG" > "$OUT/log_tail.txt"
echo "--- collected in $OUT"
