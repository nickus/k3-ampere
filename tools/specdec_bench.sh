#!/bin/bash
# Speculative decoding under pipeline parallelism, on REAL weights, on Ampere.
#
# Why this model: acceptance rate is meaningless with a random draft, and real
# Kimi-K3 weights are 837 GB (REAP-448) - they do not fit any 3090 box. GLM-4.5-Air
# ships `num_nextn_predict_layers: 1`, i.e. a real MTP head, in a 4-bit AWQ build
# of ~63 GB that fits 8x3090 and runs through Marlin on sm_86. vLLM's V2 gate
# blocks only `eagle3 + PP>1`, so `mtp` under PP is reachable.
#
# What it measures, per configuration, with repeats:
#   - mean acceptance length (the mechanism-level number vllm#50514 asks for)
#   - output tokens/s and TPOT (the number we owe)
# Configurations: PP=4 and PP=8, spec off and on. PP=1 is impossible here - the
# 60 GB model does not fit one 24 GB card, which is the whole reason this rig is
# pipeline-parallel. Both degrees are > 2, i.e. exactly the region vllm#50514
# caps itself at and calls unvalidated on hardware.
set -u
cd /workspace/k3
MODEL=${MODEL:-/workspace/models/glm45air}
PORT=18300
REPS=${REPS:-5}
OUT=/workspace/k3/bench
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

serve() {  # serve <pp> <spec yes|no> <k>
  pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 8
  local args=(--model "$MODEL" --served-model-name m --trust-remote-code
    --pipeline-parallel-size "$1" --tensor-parallel-size 1
    --enable-prefix-caching --max-model-len 8192 --no-async-scheduling
    --gpu-memory-utilization 0.90 --enforce-eager --port $PORT)
  [ "$2" = yes ] && args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$3}")
  nohup /venv/nm/bin/python -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$OUT/srv_pp$1_$2$3.log" 2>&1 &
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && return 0
  done
  return 1
}

# One measurement pass: fixed prompts, greedy, fixed max_tokens.
bench() {  # bench <tag>
  PYTHONPATH= /venv/nm/bin/python - "$PORT" "$1" "$OUT" <<'PY'
import json, sys, time, urllib.request

port, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a race condition is, with a short example.",
    "Refactor this loop to use a dict comprehension: for k in keys: d[k] = f(k)",
    "Write a SQL query returning the top 5 customers by total order value.",
    "What does the Linux OOM killer do and how do you tune it?",
    "Implement binary search over a rotated sorted array in Go.",
    "Describe the difference between a process and a thread.",
    "Write a bash one-liner that finds the 10 largest files under /var.",
]
rows = []
t_all = time.perf_counter()
for i, p in enumerate(PROMPTS):
    body = json.dumps({
        "model": "m", "prompt": p, "max_tokens": 128, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", body,
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    n = d.get("usage", {}).get("completion_tokens", 0)
    rows.append({"i": i, "sec": dt, "out_tokens": n, "tok_s": n / dt if dt else 0})
wall = time.perf_counter() - t_all
total = sum(r["out_tokens"] for r in rows)
rec = {"tag": tag, "wall_s": wall, "total_out_tokens": total,
       "agg_tok_s": total / wall if wall else 0, "per_request": rows}
with open(f"{out}/bench.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"{tag}: {total} tok in {wall:.1f}s -> {rec['agg_tok_s']:.1f} tok/s")
PY
}

acceptance() {  # acceptance <logfile> -> last reported mean acceptance length
  grep -o "Mean acceptance length: [0-9.]*" "$1" | tail -1 | awk '{print $4}'
}

for PP in 4 8; do
  for SPEC in no yes; do
    K=3; [ "$SPEC" = no ] && K=0
    TAG="pp${PP}_spec${SPEC}"
    if ! serve "$PP" "$SPEC" "$K"; then
      echo "$TAG: NEVER_SERVED"; grep -iE "error|Traceback" "$OUT/srv_pp${PP}_${SPEC}${K}.log" | head -3 | cut -c1-160
      continue
    fi
    bench "${TAG}_warmup" >/dev/null 2>&1   # discard the first pass (JIT, caches)
    for r in $(seq 1 "$REPS"); do bench "${TAG}_r${r}"; done
    A=$(acceptance "$OUT/srv_pp${PP}_${SPEC}${K}.log")
    echo "$TAG: mean acceptance length = ${A:-n/a}"
  done
done
pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null
echo BENCH_DONE
