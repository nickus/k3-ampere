#!/bin/bash
# Three gates for the DSpark-under-PP patch. Run on a 2x3090 box after
# tools/dspark_pp_patch.py has been applied to the installed vLLM.
#
#   Gate 1  boots at PP=2 (Problem A handled)
#   Gate 2  taps identical at PP=1 and PP=2 (Problem B handled) <- the real test
#   Gate 3  drafts actually accepted, output still correct
#
# Gate 2 is not optional: Problem B fails silently, so "it boots and generates"
# is not evidence of anything.
cd /workspace/k3
PY=/venv/nm/bin/python
export PYTHONPATH=/workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DSPARK_PROBE=1

cat > sitecustomize.py <<'EOF'
try:
    import dspark_pp_probe  # noqa: F401
except Exception as e:  # never let instrumentation break the server
    print("[DSPARK_PROBE] not installed:", e, flush=True)
EOF

SPEC='{"model":"/workspace/k3/k3-dspark-draft","method":"dspark","num_speculative_tokens":3}'

serve() {  # serve <tag> <pp>
  local tag="$1" pp="$2"
  pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 5
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code \
    --load-format dummy --pipeline-parallel-size "$pp" --tensor-parallel-size 1 \
    --speculative-config "$SPEC" \
    --max-model-len 4096 --gpu-memory-utilization 0.82 --enforce-eager \
    --port 18000 > "srv_$tag.log" 2>&1 &
  for i in $(seq 1 50); do
    curl -s http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q '"id"' && return 0
    grep -qiE "NotImplementedError|AssertionError|RuntimeError|Error response" "srv_$tag.log" && return 1
    sleep 10
  done
  return 1
}

ask() {
  curl -s http://127.0.0.1:18000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"k3","prompt":"The quick brown fox jumps over the lazy dog and then","max_tokens":24,"temperature":0}' \
    | $PY -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null
}

echo "##### GATE 1: boot at PP=2"
if serve pp2 2; then echo "GATE1 PASS"; else
  echo "GATE1 FAIL"; grep -iE "NotImplementedError|Error|assert" srv_pp2.log | tail -5 | cut -c1-200; exit 1
fi
OUT_PP2=$(ask); grep "DSPARK_PROBE.*shape" srv_pp2.log | head -3 > probe_pp2.txt
echo "  taps seen: $(grep -c 'DSPARK_PROBE.*shape' srv_pp2.log)"
pkill -9 -f "[a]pi_server" 2>/dev/null; sleep 5

echo "##### reference: PP=1"
if serve pp1 1; then echo "PP1 boots"; else echo "PP1 FAIL"; exit 1; fi
OUT_PP1=$(ask); grep "DSPARK_PROBE.*shape" srv_pp1.log | head -3 > probe_pp1.txt
pkill -9 -f "[a]pi_server" 2>/dev/null

echo "##### GATE 2: tap fingerprints must match"
if diff -q probe_pp1.txt probe_pp2.txt >/dev/null 2>&1; then
  echo "GATE2 PASS - taps identical across PP"
else
  echo "GATE2 FAIL - taps differ:"; echo "--- PP=1:"; cat probe_pp1.txt; echo "--- PP=2:"; cat probe_pp2.txt
fi

echo "##### GATE 3: output parity + drafts accepted"
[ "$OUT_PP1" = "$OUT_PP2" ] && echo "GATE3a PASS - identical text" || {
  echo "GATE3a FAIL"; echo "  PP1: $OUT_PP1"; echo "  PP2: $OUT_PP2"; }
echo "  acceptance counters (PP=2):"
grep -iE "accept|draft" srv_pp2.log | grep -viE "error|warn" | tail -3 | cut -c1-180
echo DSPARK_PP_TEST_DONE
