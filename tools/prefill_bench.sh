#!/bin/bash
# Measure PREFILL honestly.
#
# Every TTFT recorded so far in this session is a prefix-cache hit: caching was on
# and the same eight prompts repeated, so ~0.09 s was the cost of looking up a
# cached prefix, not of computing one. This disables prefix caching and sends
# prompts that are unique per request, at several lengths, so TTFT is prefill.
#
# Reports tokens/s of prefill as a function of prompt length, because prefill is
# compute-bound and the rate is only meaningful against a length.
set -u
cd /workspace/k3
MODEL=${MODEL:-/workspace/models/glm45air}
PORT=18600
PP=${PP:-4}
OUT=/workspace/k3/bench
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

pkill -9 -f "[a]pi_server" 2>/dev/null; sleep 5
nohup /venv/nm/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name ${NAME:-m} --trust-remote-code \
  --pipeline-parallel-size "$PP" --tensor-parallel-size 1 \
  --no-enable-prefix-caching --max-model-len ${MAXLEN:-8192} --no-async-scheduling \
  --max-num-seqs ${MAXSEQS:-16} \
  --gpu-memory-utilization ${UTIL:-0.90} --enforce-eager --port $PORT \
  > "$OUT/prefill_srv_${NAME:-m}.log" 2>&1 &

for i in $(seq 1 90); do
  sleep 10
  curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && break
done

/venv/nm/bin/python - "$PORT" "$OUT" "${LENGTHS:-256,512,1024,2048,4096}" "${NAME:-m}" <<'PY'
import json, random, sys, time, urllib.request

port, out = sys.argv[1], sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3].split(",")]
NAME = sys.argv[4]
# Unique text per request so nothing can be served from any cache, and long
# enough that TTFT is dominated by prefill rather than by scheduling.
WORDS = ["system", "kernel", "buffer", "thread", "socket", "commit", "branch",
         "vector", "matrix", "tensor", "packet", "daemon", "cursor", "handle"]
rng = random.Random(20260812)

rows = []
for target in LENGTHS:
    for rep in range(3):
        prompt = " ".join(rng.choice(WORDS) for _ in range(int(target * 0.75)))
        body = json.dumps({
            "model": NAME, "prompt": prompt, "max_tokens": 1, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", body,
            {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        ttft = None
        usage = None
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                s = raw.decode().strip()
                if not s.startswith("data: "):
                    continue
                p = s[6:]
                if p == "[DONE]":
                    break
                d = json.loads(p)
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0]
                if ch.get("text") and ttft is None:
                    ttft = time.perf_counter() - t0
        n_in = (usage or {}).get("prompt_tokens", 0)
        rows.append({"target": target, "rep": rep, "prompt_tokens": n_in,
                     "ttft_s": ttft,
                     "prefill_tok_s": (n_in / ttft) if ttft else None})
        print(f"{n_in:>6} tok  TTFT {ttft * 1000:8.1f} ms  "
              f"{n_in / ttft:9.0f} tok/s prefill", flush=True)

with open(f"{out}/prefill_{NAME}.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
PY
pkill -9 -f "[a]pi_server" 2>/dev/null
echo PREFILL_DONE
