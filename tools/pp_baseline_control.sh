#!/bin/bash
# Does vLLM's pipeline path change greedy output at all, on a stock model?
#
# This control should have come first. The K3 stand diverges between PP=1 and
# PP=2 even with speculation switched off entirely, so every greedy-parity
# conclusion drawn from that stand is confounded: the K3 model file on this box
# carries our own B1-B5 patches for carrying aux taps across stage boundaries,
# and a bug there would look exactly like a spec-decode bug.
#
# Qwen3-0.6B is stock, 28 layers, untouched by anything we wrote. If PP=1 and
# PP=2 agree here, the divergence belongs to the K3 path (ours or upstream's);
# if they disagree, vLLM's pipeline parallelism is not greedy-stable and nothing
# measured on top of it means anything.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
MODEL=${MODEL:-/workspace/models/qwen06}
OUT=/workspace/k3/pp_control
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

run() {  # run <pp>
  local pp=$1 port=$((18900 + pp))
  pkill -9 -f "api_serve[r]" 2>/dev/null; sleep 5
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name q --trust-remote-code \
    --pipeline-parallel-size "$pp" --tensor-parallel-size 1 \
    --max-model-len 2048 --no-async-scheduling --max-num-seqs 1 \
    --gpu-memory-utilization 0.35 --enforce-eager --port "$port" \
    > "$OUT/srv_pp${pp}.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q '"id"' && break
  done
  curl -s --max-time 300 "http://127.0.0.1:$port/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"q","prompt":"The capital of France is","max_tokens":48,"temperature":0}' \
    > "$OUT/resp_pp${pp}.json"
  pkill -9 -f "api_serve[r]" 2>/dev/null; sleep 3
}

for pp in ${PPS:-1 2}; do run "$pp"; done

$PY - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
texts = {}
for pp in (1, 2):
    try:
        d = json.load(open(f"{out}/resp_pp{pp}.json"))
        texts[pp] = d["choices"][0]["text"]
    except Exception as e:
        texts[pp] = f"<no response: {type(e).__name__}>"
for pp, t in texts.items():
    print(f"PP={pp}: {t[:120]!r}")
print("IDENTICAL" if texts.get(1) == texts.get(2) else "DIVERGENT")
PY
