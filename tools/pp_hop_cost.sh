#!/bin/bash
# What does one pipeline stage boundary cost per token?
#
# The rig puts K3's 93 layers on ~50 stages; this box measured them on 8. Total
# compute is the same either way, so the difference is hop count: ~50 boundaries
# per token instead of 8. That difference is the single biggest unknown in any
# projection to the rig, and it is directly measurable.
#
# The 16-layer stand is the right instrument precisely because it is tiny: hidden
# 1024, so per-layer compute is negligible and decode time is dominated by the
# per-stage overhead we want to isolate. Sweep PP and read the slope.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
OUT=${OUT:-/workspace/k3/hop_cost}
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
# EAGER=1 keeps --enforce-eager (how every measurement in this repo was taken,
# inherited from the debugging harness); EAGER=0 lets vLLM capture CUDA graphs.
# The whole point of the sweep is the difference between the two slopes: at ~50
# stages the launch overhead is over half the per-token budget, and graphs are
# what removes it.
if [ "${EAGER:-1}" = 1 ]; then EAGER_ARG="--enforce-eager"; else EAGER_ARG=""; fi

for pp in ${PPS:-1 2 4 8}; do
  port=$((19200 + pp))
  pkill -9 -f "api_serve[r]" 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9
  sleep 8
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code \
    --load-format dummy --pipeline-parallel-size "$pp" --tensor-parallel-size 1 \
    --max-model-len 1024 --max-num-seqs 1 --no-enable-prefix-caching \
    --gpu-memory-utilization 0.30 ${EAGER_ARG} --port "$port" \
    > "$OUT/srv_pp${pp}.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q '"id"' && break
  done
  $PY - "$port" "$pp" "$OUT" <<'PY'
import json, sys, time, urllib.request
port, pp, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompt = "The quick brown fox jumps over the lazy dog and then"
best = None
for _ in range(3):  # take the fastest pass; we want the floor, not the average
    body = json.dumps({"model": "k3", "prompt": prompt, "max_tokens": 64,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data: "): continue
            p = s[6:]
            if p == "[DONE]": break
            d = json.loads(p)
            if d.get("usage"): usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            if ch.get("text"):
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1
    wall = time.perf_counter() - t0
    n = (usage or {}).get("completion_tokens") or n
    if n > 1:
        tpot = (wall - ttft) / (n - 1) * 1000
        best = tpot if best is None else min(best, tpot)
print(f"PP={pp}: TPOT {best:.2f} ms" if best else f"PP={pp}: no data", flush=True)
with open(f"{out}/hops.jsonl", "a") as f:
    f.write(json.dumps({"pp": pp, "tpot_ms": best, "eager": __import__("os").environ.get("EAGER","1")}) + "\n")
PY
done
pkill -9 -f "api_serve[r]" 2>/dev/null

$PY - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/hops.jsonl")]
rows = [r for r in rows if r["tpot_ms"]]
rows.sort(key=lambda r: r["pp"])
for r in rows:
    print(f"  PP={r['pp']:>2}  {r['tpot_ms']:7.2f} ms")
if len(rows) >= 2:
    # Fit tpot = a + b*pp by least squares; b is the per-stage cost.
    n = len(rows)
    sx = sum(r["pp"] for r in rows); sy = sum(r["tpot_ms"] for r in rows)
    sxx = sum(r["pp"] ** 2 for r in rows); sxy = sum(r["pp"] * r["tpot_ms"] for r in rows)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    print(f"\nfit: TPOT = {a:.2f} + {b:.2f} * pp   (ms)")
    print(f"per added stage: {b:.2f} ms/token")
    for target in (30, 50):
        print(f"  projected stage overhead at PP={target}: {b * target:.0f} ms/token")
PY
