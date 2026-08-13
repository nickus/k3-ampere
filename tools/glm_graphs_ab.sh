#!/bin/bash
# eager vs CUDA graphs on a REAL model, which is the half the stand cannot price.
#
# The stand sweep priced the per-stage term (3.03 -> 1.82 ms) and showed the fixed
# term collapse to nothing — but a stand with negligible per-layer arithmetic says
# nothing about whether graphs also speed up real model work. GLM-4.5-Air AWQ is a
# real 60 GB model with 93-layer-class kernel counts, and it fits this box.
#
# Two configurations, nothing else varied: --enforce-eager on and off, PP=4,
# concurrency 1, decode TPOT. The ratio is the number the rig projection needs.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
MODEL=${MODEL:-/workspace/models/glm45air}
PORT=19500
OUT=/workspace/k3/bench
mkdir -p "$OUT"
rm -f "$OUT/glm_ab.jsonl"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

run() {  # run <tag> <eager 1|0>
  local tag=$1 eager=$2
  pkill -9 -f "api_serve[r]" 2>/dev/null
  pkill -9 -f "VLL[M]::" 2>/dev/null
  sleep 20
  local args=(--model "$MODEL" --served-model-name m --trust-remote-code
    --pipeline-parallel-size 4 --tensor-parallel-size 1
    ${SPEC_ARG:-}
    --max-model-len 4096 --max-num-seqs 8 --no-enable-prefix-caching
    --gpu-memory-utilization 0.88 --port $PORT)
  [ "$eager" = 1 ] && args+=(--enforce-eager)
  nohup $PY -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$OUT/glm_$tag.log" 2>&1 &
  # Graph capture adds minutes on top of a 60 GB load; allow 25.
  for i in $(seq 1 150); do
    sleep 10
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && break
    grep -qiE "ValueError|OutOfMemory|NotImplementedError" "$OUT/glm_$tag.log" 2>/dev/null && \
      { echo "$tag: early failure"; return 1; }
  done
  $PY /workspace/k3/bench_client.py --port $PORT --tag "$tag" \
    --out "$OUT/glm_ab.jsonl" --model-name m --concurrency 1 \
    --max-tokens 256 --reps 2
  pkill -9 -f "api_serve[r]" 2>/dev/null
  pkill -9 -f "VLL[M]::" 2>/dev/null
  sleep 15
}

# SPEC=on adds the model's own MTP head. Graphs and speculation together are the
# combination the rig actually needs, and the one most likely to silently fall
# back to eager: the verify step's batch shape depends on how many drafts were
# accepted, and a captured graph is a fixed shape.
if [ "${SPEC:-off}" = on ]; then
  SPEC_ARG='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
fi
run ${TAGPREFIX:-}eager 1
run ${TAGPREFIX:-}graphs 0

$PY - "$OUT" <<'PY'
import json, sys, collections, statistics
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/glm_ab.jsonl")]
rows = [r for r in rows if r["median_tpot_ms"]]
by = collections.defaultdict(list)
for r in rows:
    by[r["tag"]].append(r["median_tpot_ms"])
for t, v in sorted(by.items()):
    print(f"{t:8s}: " + ", ".join(f"{x:.2f}" for x in v) + " ms")
if "eager" in by and "graphs" in by:
    e, g = statistics.mean(by["eager"]), statistics.mean(by["graphs"])
    print(f"\nGLM-4.5-Air PP=4: eager {e:.2f} ms -> graphs {g:.2f} ms   {e/g:.2f}x")
    print(f"  per-token saving: {e-g:.1f} ms")
PY
echo GLM_AB_DONE
