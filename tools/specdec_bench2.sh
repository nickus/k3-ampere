#!/bin/bash
# Speculative decoding under pipeline parallelism: the defensible version.
#
# v1 reported aggregate pass tokens/s at batch 1 with prefix caching on. Both
# choices flatter speculation, and both are the first things a reviewer attacks:
#   - cached prefill deletes the one component speculation cannot accelerate;
#   - batch 1 under PP leaves pp_size-1 stages idle, and speculation partially
#     fills bubbles that real concurrency fills anyway - which is the whole
#     reason anyone runs a pipeline.
# So v2 measures decode TPOT (prefill excluded by construction) and sweeps
# concurrency until the pipeline is actually full.
# NOTE: no --no-async-scheduling. Only AsyncScheduler assigns
# next_decode_eligible_step, which is what keeps a request's decodes pp_size
# apart under PP; the base scheduler reads that field and never sets it. With the
# flag on, spec decode under PP reads the sampled-token ring out of phase.
set -u
cd /workspace/k3
MODEL=${MODEL:-/workspace/models/glm45air}
PORT=18300
REPS=${REPS:-2}
MAXTOK=${MAXTOK:-512}
CONCS=${CONCS:-"1 4 8"}
PPS=${PPS:-"4 8"}
OUT=/workspace/k3/bench
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

serve() {  # serve <pp> <spec yes|no> <k>
  pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 8
  local args=(--model "$MODEL" --served-model-name m --trust-remote-code
    --pipeline-parallel-size "$1" --tensor-parallel-size 1
    --enable-prefix-caching --max-model-len 8192
    --max-num-seqs 16
    --gpu-memory-utilization 0.90 --enforce-eager --port $PORT)
  [ "$2" = yes ] && args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$3}")
  nohup /venv/nm/bin/python -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$OUT/v2_srv_pp$1_$2$3.log" 2>&1 &
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && return 0
  done
  return 1
}

for PP in $PPS; do
  for SPEC in no yes; do
    K=3; [ "$SPEC" = no ] && K=0
    TAG="pp${PP}_spec${SPEC}"
    if ! serve "$PP" "$SPEC" "$K"; then
      echo "$TAG: NEVER_SERVED"
      grep -iE "error|Traceback" "$OUT/v2_srv_pp${PP}_${SPEC}${K}.log" | head -3 | cut -c1-160
      continue
    fi
    for C in $CONCS; do
      /venv/nm/bin/python /workspace/k3/bench_client.py \
        --port $PORT --tag "$TAG" --out "$OUT/bench2.jsonl" \
        --concurrency "$C" --max-tokens "$MAXTOK" --reps "$REPS"
    done
    # Acceptance length is only meaningful with the bonus token accounted for;
    # vLLM's own figure includes it, which is what we quote.
    A=$(grep -o "Mean acceptance length: [0-9.]*" "$OUT/v2_srv_pp${PP}_${SPEC}${K}.log" | tail -1 | awk '{print $4}')
    echo "$TAG: mean acceptance length = ${A:-n/a}"
    echo "{\"tag\":\"$TAG\",\"acceptance\":\"${A:-}\"}" >> "$OUT/acceptance.jsonl"
  done
done
pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null
echo BENCH2_DONE
