#!/bin/bash
# Is the slice's greedy output stable against batch shape, with no speculation?
#
# One K3 prompt in eight differs between spec-on and spec-off. Before calling
# that a speculative-decoding defect, rule out the boring explanation: a
# 24-of-448 expert slice produces degenerate, low-confidence text, and when the
# top two logits are nearly tied, any change in batch shape can flip the argmax.
# Speculation changes batch shape by construction - it verifies k+1 positions at
# once - so a tie-flip would look exactly like a spec bug.
#
# This runs the SAME server with speculation OFF and only varies concurrency,
# which changes batch shape and nothing else. If a prompt's text changes here,
# the model is tie-sensitive and the 1-in-8 divergence is not evidence about
# speculation.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
MODEL=${MODEL:-/workspace/models/k3-slice}
PORT=18800
OUT=/workspace/k3/bench
mkdir -p "$OUT"
rm -f "$OUT/shape.jsonl"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-11,13,13,13,13,13,13,4}

pkill -9 -f "api_serve[r]" 2>/dev/null; sleep 5
nohup $PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name k3 --trust-remote-code \
  --pipeline-parallel-size 8 --tensor-parallel-size 1 \
  --max-model-len 384 --max-num-seqs 8 --no-enable-prefix-caching \
  --gpu-memory-utilization 0.95 --enforce-eager --port $PORT \
  > "$OUT/shape_srv.log" 2>&1 &

for i in $(seq 1 240); do
  sleep 10
  curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && break
  grep -qiE "Traceback|Error" "$OUT/shape_srv.log" 2>/dev/null && { echo "SERVER FAILED"; exit 1; }
done

for c in 1 4; do
  $PY /workspace/k3/bench_client.py --port $PORT --tag "shape_c${c}" \
    --out "$OUT/shape.jsonl" --model-name k3 --concurrency "$c" \
    --max-tokens 256 --reps 1
done
pkill -9 -f "api_serve[r]" 2>/dev/null

$PY - "$OUT" <<'PY'
import json, sys, collections
out = sys.argv[1]
h = collections.defaultdict(dict)
for line in open(f"{out}/shape.jsonl"):
    d = json.loads(line)
    for r in d["per_request"]:
        h[r["prompt_idx"]][d["tag"]] = (r["text_sha"], r["text_head"])
bad = [k for k, v in h.items() if len({s for s, _ in v.values()}) > 1]
print(f"prompts whose text changed with batch shape alone: {len(bad)}/{len(h)} {bad}")
for k in bad:
    for tag, (_, head) in sorted(h[k].items()):
        print(f"  prompt {k} {tag}: {head[:70]!r}")
PY
