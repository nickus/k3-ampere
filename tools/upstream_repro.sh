#!/bin/bash
# Minimal upstream reproduction: stock model, stock vLLM, no patches of ours.
#
# The claim to test is that speculative decoding under pipeline parallelism gives
# different greedy output with `--no-async-scheduling` than without, because only
# AsyncScheduler assigns `next_decode_eligible_step` - the field the base
# scheduler reads at scheduler.py:509 to keep a request's decodes pp_size apart.
# PP=4, not 2: the 60 GB AWQ target does not fit on two 24 GB cards.
#
# Everything Kimi-specific is out of the picture here:
#   * GLM-4.5-Air AWQ with its own MTP head - a stock model, stock method
#   * a separate venv with an unpatched nightly, carrying exactly ONE line of
#     ours: the draft-model PP verification relaxation from issue 1, without
#     which MTP under PP cannot start at all. That dependency is itself the
#     point: issue 1 hides issue 2.
#
# Four runs: {async on, async off} x {spec off, spec on}. The control matters as
# much as the test - if spec-off already differs between the two scheduler modes,
# the finding is about the scheduler generally and not about speculation.
set -u
PY=/venv/clean/bin/python
MODEL=${MODEL:-/workspace/models/glm45air}
OUT=/workspace/k3/upstream_repro
mkdir -p "$OUT"

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

run() {  # run <tag> <spec on|off> <async on|off>
  local tag=$1 spec=$2 async=$3 port=19100
  # Killing by name leaves the worker processes holding VRAM, and the next run
  # then fails with "Free memory on device cuda:0 (2.96/23.56 GiB)" - which looks
  # like a startup bug and is really the previous run refusing to die. Kill
  # whoever actually holds GPU memory.
  pkill -9 -f "api_serve[r]" 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9
  sleep 10
  local args=(--model "$MODEL" --served-model-name q --trust-remote-code
    --pipeline-parallel-size 4 --tensor-parallel-size 1
    --max-model-len 2048 --max-num-seqs 4 --no-enable-prefix-caching
    --gpu-memory-utilization 0.90 --enforce-eager --port $port)
  # The V2 runner rejects both `ngram` and the plain draft-model method
  # ("Model Runner V2 does not yet support ..."), so the only stock route to
  # speculation here is the model's own MTP head.
  [ "$spec" = on ] && args+=(--speculative-config \
    '{"method":"mtp","num_speculative_tokens":3}')
  [ "$async" = off ] && args+=(--no-async-scheduling)
  nohup $PY -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$OUT/srv_$tag.log" 2>&1 &
  # 15 minutes: a 60 GB AWQ target across 4 stages takes well over the 5 the
  # first version allowed, and a wait that expires early looks exactly like a
  # server that refused to start.
  for i in $(seq 1 90); do
    sleep 10
    curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q '"id"' && break
    grep -qiE "ValueError|NotImplementedError|OutOfMemory" "$OUT/srv_$tag.log" 2>/dev/null && \
      { echo "$tag: early failure"; break; }
  done
  # Short, deterministic, greedy.
  curl -s --max-time 300 "http://127.0.0.1:$port/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"q","prompt":"List the colors: red, green, blue, red, green, blue, red, green,","max_tokens":64,"temperature":0}' \
    > "$OUT/resp_$tag.json"
  pkill -9 -f "api_serve[r]" 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9
  sleep 8
}

run async_on_spec_off  off on
run async_on_spec_on   on  on
run async_off_spec_off off off
run async_off_spec_on  on  off

$PY - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
t = {}
for tag in ("async_on_spec_off", "async_on_spec_on",
            "async_off_spec_off", "async_off_spec_on"):
    try:
        t[tag] = json.load(open(f"{out}/resp_{tag}.json"))["choices"][0]["text"]
    except Exception as e:
        t[tag] = f"<no response: {type(e).__name__}>"
for k, v in t.items():
    print(f"{k:22s}: {v[:90]!r}")
print()
print("spec on vs off, async ON :",
      "IDENTICAL" if t["async_on_spec_on"] == t["async_on_spec_off"] else "DIVERGENT")
print("spec on vs off, async OFF:",
      "IDENTICAL" if t["async_off_spec_on"] == t["async_off_spec_off"] else "DIVERGENT")
print("control, spec off, async on vs off:",
      "IDENTICAL" if t["async_on_spec_off"] == t["async_off_spec_off"] else "DIVERGENT")
PY
