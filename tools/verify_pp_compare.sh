#!/bin/bash
# Compare the target's VERIFICATION logits between two PP degrees.
#
# This is the instrument that proved the relay at PP=2 against PP=1, and it is
# weight-independent: it compares the target's verification of the drafts
# against itself across PP degrees, so a random-weight stand is fine. Greedy text
# parity cannot do that here - on this stand no draft is ever accepted, so
# spec-on and spec-off produce identical text whether or not the drafts arrived.
#
# Real weights at PP=8 showed drafts arriving as zeros (token 0 = '!' all over
# the output). This reproduces that on a stand that loads in seconds instead of
# twelve minutes, by running the SAME prompt at two PP degrees and diffing the
# per-step traces.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
OUT=/workspace/k3/verify_traces
mkdir -p "$OUT"

export PYTHONPATH=/workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# SPEC=off runs the same stand with no draft at all. Without this control a
# PP-vs-PP text difference cannot be attributed to speculative decoding rather
# than to the pipeline path itself.
if [ "${SPEC:-on}" = off ]; then
  SPEC_ARGS=""
else
  SPEC_ARGS='--speculative-config {"method":"dspark","model":"/workspace/k3/k3-dspark-draft","num_speculative_tokens":3}'
fi
# Optional: make the illegal-access report land on the kernel that actually
# faulted instead of on the next synchronising op.
[ -n "${BLOCKING:-}" ] && export CUDA_LAUNCH_BLOCKING=1

run_one() {  # run_one <pp>
  local pp=$1
  local port=$((18700 + pp))
  local log="$OUT/srv_pp${pp}.log"
  pkill -9 -f "[a]pi_server" 2>/dev/null; sleep 5
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code \
    --pipeline-parallel-size "$pp" --tensor-parallel-size 1 \
    --load-format dummy --max-model-len 2048 --no-async-scheduling --max-num-seqs 1 \
    --gpu-memory-utilization 0.60 --enforce-eager --port "$port" \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"/workspace/k3/k3-dspark-draft\",\"num_speculative_tokens\":3}" \
    > "$log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q '"id"' && break
  done
  # NREQ concurrent requests. With one request in flight and pp_size stages, the
  # scheduler has nothing else to fill the pipeline with, so it re-schedules the
  # SAME request before its sampled token has come home - which is precisely the
  # state where the anchor reads as zero. More requests should make each one land
  # in a step at most once per pp_size steps.
  for r in $(seq 1 ${NREQ:-1}); do
    curl -s --max-time 300 "http://127.0.0.1:$port/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"k3\",\"prompt\":\"Request $r: the quick brown fox jumps over the lazy dog and then\",\"max_tokens\":24,\"temperature\":0}" \
      > "$OUT/resp_pp${pp}_r${r}.json" &
  done
  wait
  cp "$OUT/resp_pp${pp}_r1.json" "$OUT/resp_pp${pp}.json" 2>/dev/null || true
  pkill -9 -f "[a]pi_server" 2>/dev/null; sleep 3
  # Strip the rank/pid prefix: it differs by construction and comparing whole
  # lines made this gate impossible to pass at PP>1 once already.
  grep -o '\[VERIFY\].*' "$log" > "$OUT/trace_pp${pp}.txt" || true
  echo "PP=$pp: $(wc -l < "$OUT/trace_pp${pp}.txt") verify lines"
}

for pp in ${PPS:-2 8}; do run_one "$pp"; done

A=$OUT/trace_pp$(echo ${PPS:-2 8} | awk '{print $1}').txt
B=$OUT/trace_pp$(echo ${PPS:-2 8} | awk '{print $2}').txt
echo "=== first 6 lines each"
head -6 "$A"; echo "---"; head -6 "$B"
echo "=== diff"
if [ ! -s "$A" ] || [ ! -s "$B" ]; then
  echo "TRACE MISSING - probe did not fire; nothing is being compared"
elif diff -q "$A" "$B" >/dev/null; then
  echo "IDENTICAL across PP degrees ($(wc -l < "$A") lines)"
else
  echo "DIVERGENT"
  diff "$A" "$B" | head -12
fi
