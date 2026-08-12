#!/bin/bash
# Kimi-K3, 24 experts of 448, with the REAL DSpark draft, on 8x3090.
#
# What this is for: every spec-decode result so far was measured either on a
# dummy 4-layer stand (weights random, acceptance meaningless) or on GLM-4.5-Air
# (real, but a different architecture and an MTP head rather than DSpark). This
# runs the actual target architecture - 93 layers, KDA linear attention, MLA,
# MXFP4 - with the actual draft checkpoint, under real pipeline parallelism.
#
# What it is NOT: a quality or acceptance-rate result for Kimi-K3. The draft was
# trained against the full 448-expert target; against a 24-expert one it is
# predicting a different distribution. A crippled MoE also tends to degenerate
# into low-entropy repetition under greedy, which ANY draft predicts easily - so
# a HIGH acceptance number here would be the least trustworthy outcome of all.
# That is why this dumps sample generations next to the number: a reader has to
# be able to see the degeneracy for themselves.
set -u
cd /workspace/k3
MODEL=${MODEL:-/workspace/models/k3-slice}
DRAFT=${DRAFT:-/workspace/models/k3-dspark-rh}
PORT=18400
PP=${PP:-8}
K=${K:-1}
OUT=/workspace/k3/bench
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# The draft lives on the LAST pipeline stage and at an even 93/8 split that
# stage has no room for it - measured OOM creating draft weights twice.
# Budget per stage, from the measured 141.3 GB over 93 layers (1.52 GB/layer):
#   stage 0    : L*1.52 + 2.35 (embedding)
#   stages 1-6 : L*1.52
#   stage 7    : L*1.52 + 2.35 (lm_head) + 4.73 (draft)
# Two attempts (9 then 8 layers on the last stage) still OOMed, and the second
# gave the missing number: 23.01 GB in use with only 8 layers, i.e. the draft
# costs ~9.7 GB resident, not the 4.73 GB of its checkpoint - it carries its own
# 163840-row embedding and lm_head on top of the file. Re-solving with that:
#   stage 7 <= 22 GB  =>  L*1.52 + 2.35 + 9.7 + 1 <= 22  =>  L <= 5
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-11,13,13,13,13,13,12,5}

serve() {  # serve <spec yes|no>
  pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 8
  local args=(--model "$MODEL" --served-model-name k3 --trust-remote-code
    --pipeline-parallel-size "$PP" --tensor-parallel-size 1
    --max-model-len 512 --no-async-scheduling --max-num-seqs 1
    --gpu-memory-utilization 0.98 --enforce-eager --port $PORT)
  [ "$1" = yes ] && args+=(--speculative-config \
    "{\"method\":\"dspark\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K}")
  nohup /venv/nm/bin/python -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$OUT/k3_srv_$1.log" 2>&1 &
  # 152 GB off local disk across 8 stages: allow 40 minutes before giving up.
  for i in $(seq 1 240); do
    sleep 10
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && return 0
    grep -qiE "Traceback|Error" "$OUT/k3_srv_$1.log" 2>/dev/null && \
      { echo "  early failure detected"; return 1; }
  done
  return 1
}

# Show what the model actually writes. On a 24-of-448 expert slice this is the
# single most informative output in the whole run.
sample() {  # sample <tag>
  /venv/nm/bin/python - "$PORT" "$1" <<'PY'
import json, sys, urllib.request
port, tag = sys.argv[1], sys.argv[2]
for p in ["Write a Python function that merges two sorted lists.",
          "Explain what a race condition is."]:
    body = json.dumps({"model": "k3", "prompt": p, "max_tokens": 96,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    print(f"--- {tag} | {p}\n{d['choices'][0]['text']!r}\n", flush=True)
PY
}

# spec FIRST: it is the phase that can fail, and a 12-minute load of
# 141 GB is too expensive to spend on the baseline before knowing.
for SPEC in ${ORDER:-yes no}; do
  echo "=== k3-slice spec=$SPEC pp=$PP"
  if ! serve "$SPEC"; then
    echo "k3_spec$SPEC: NEVER_SERVED"
    grep -iE "error|Traceback|NotImplemented" "$OUT/k3_srv_$SPEC.log" | head -5 | cut -c1-200
    continue
  fi
  sample "k3_spec$SPEC" | tee -a "$OUT/k3_samples.txt"
  /venv/nm/bin/python /workspace/k3/bench_client.py --port $PORT \
    --tag "k3pp${PP}_spec${SPEC}" --out "$OUT/k3_bench.jsonl" \
    --model-name k3 --concurrency 1 --max-tokens 256 --reps 2
  A=$(grep -o "Mean acceptance length: [0-9.]*" "$OUT/k3_srv_$SPEC.log" | tail -1 | awk '{print $4}')
  echo "k3 spec=$SPEC: mean acceptance length = ${A:-n/a}"
done
pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null
echo K3_SLICE_DONE
