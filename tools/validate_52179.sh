#!/bin/bash
# Validate vllm PR #52179 (PP spec-decode cadence with sync scheduling) at PP=4
# with the DSpark method — the second speculative method and a deeper pipeline
# than the author's 2-GPU MTP validation, which is exactly what he asked for.
#
# Protocol (three serves on the 16-layer stand, ~1.5 min each):
#   A. async scheduling ON,  no #52179  -> reference text (known-good config)
#   B. --no-async-scheduling, no #52179 -> must DIFFER / corrupt (bug present)
#   C. --no-async-scheduling, + #52179  -> must EQUAL the reference
#
# PP=1 is deliberately not the reference: on an earlier nightly the PP=1
# no-async+spec stand died of an unrelated engine error and burned the budget.
# The PR's claim is "sync now behaves like async", so async-ON is the honest
# baseline anyway.
set -u
cd /workspace/k3
PY=/venv/nm/bin/python
OUT=/workspace/k3/v52179
mkdir -p "$OUT"
rm -f "$OUT"/resp_*.json

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0

run() {  # run <tag> <noasync 0|1>
  local tag=$1 noasync=$2 port=19700
  pkill -9 -f "api_serve[r]" 2>/dev/null; pkill -9 -f "VLL[M]::" 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9
  sleep 12
  local args=(--model /workspace/k3/k3-slice-hf --served-model-name k3
    --trust-remote-code --load-format dummy
    --pipeline-parallel-size 4 --tensor-parallel-size 1
    --max-model-len 2048 --max-num-seqs 2 --no-enable-prefix-caching
    --gpu-memory-utilization 0.4 --enforce-eager --port $port
    --speculative-config "{\"method\":\"dspark\",\"model\":\"/workspace/k3/k3-dspark-draft\",\"num_speculative_tokens\":3}")
  [ "$noasync" = 1 ] && args+=(--no-async-scheduling)
  nohup $PY -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$OUT/srv_$tag.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -q '"id"' && break
    grep -qiE "ValueError|NotImplementedError|OutOfMemory" "$OUT/srv_$tag.log" && \
      { echo "$tag: EARLY FAILURE"; return 1; }
  done
  for p in "The quick brown fox jumps over the lazy dog and then" \
           "List the colors: red, green, blue, red, green," \
           "def merge(a, b): return"; do
    curl -s --max-time 240 "http://127.0.0.1:$port/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"k3\",\"prompt\":\"$p\",\"max_tokens\":32,\"temperature\":0}" \
      >> "$OUT/resp_$tag.json"
    echo >> "$OUT/resp_$tag.json"
  done
  pkill -9 -f "api_serve[r]" 2>/dev/null; pkill -9 -f "VLL[M]::" 2>/dev/null
  echo "$tag: done"
}

run A_async_ref 0
run B_sync_bug  1

echo "=== applying PR #52179 to the installed tree"
SPEC=$($PY -c 'import vllm.v1.core.sched.scheduler as m; print(m.__file__)')
cd "$(dirname "$($PY -c 'import vllm; print(vllm.__file__)')")/.." || exit 1
git apply --include='vllm/v1/core/sched/scheduler.py' /workspace/k3/pr52179.diff \
  && echo "patch applied" || { echo "PATCH FAILED"; exit 1; }
cd /workspace/k3

run C_sync_fix 1

$PY - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
def texts(tag):
    r = []
    for line in open(f"{out}/resp_{tag}.json"):
        line = line.strip()
        if not line:
            continue
        try:
            r.append(json.load(open("/dev/null")) if False else json.loads(line)["choices"][0]["text"])
        except Exception:
            r.append(f"<unparsable: {line[:60]}>")
    return r
A, B, C = texts("A_async_ref"), texts("B_sync_bug"), texts("C_sync_fix")
for i, (a, b, c) in enumerate(zip(A, B, C)):
    print(f"prompt {i}: ref==bug: {a==b}   ref==fix: {a==c}")
    if a != c:
        print(f"  ref: {a[:70]!r}")
        print(f"  fix: {c[:70]!r}")
print()
print("VERDICT:",
      "FIX VALIDATED (sync now matches async; bug run differed)" if
      (A == C and A != B) else
      "FIX VALIDATED, bug did not manifest in run B (still ref==fix)" if A == C else
      "FIX DOES NOT RESTORE PARITY")
PY
echo V52179_DONE
