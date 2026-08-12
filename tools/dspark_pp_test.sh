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
# PP degree under test. PP=2 has no MIDDLE rank, so it never exercises the
# receive-and-forward branch this patch adds — that needs PP>=3, which is what
# 45 of 47 ranks do on the target rig. Pass PPN=4 on a 4-GPU box.
PPN=${PPN:-2}
export PYTHONPATH=/workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DSPARK_PROBE=1
# vLLM's optional usage telemetry calls py-cpuinfo, which can raise
# JSONDecodeError inside a forked worker and take the whole engine down.
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# FlashInfer's sampler is JIT-built on first use, on the LAST PP rank only,
# while every other rank sits blocked in pp_handler.receive. On an image
# without nvcc/ninja the build cannot even start (FileNotFoundError: 'ninja')
# and the whole engine dies inside warmup; with a toolchain present it is a
# multi-minute compile that is indistinguishable from a deadlock. The torch
# sampler is correct and needs no toolchain.
export VLLM_USE_FLASHINFER_SAMPLER=0

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
    --enable-prefix-caching --block-size 512 \
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
    | PYTHONPATH= $PY -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'])" 2>/dev/null
}

echo "##### GATE 1: boot at PP=$PPN"
if serve pp2 "$PPN"; then echo "GATE1 PASS"; else
  echo "GATE1 FAIL"; grep -iE "NotImplementedError|Error|assert" srv_pp2.log | tail -5 | cut -c1-200; exit 1
fi
MARK_PP2=$(grep -c "DSPARK_PROBE.*shape" srv_pp2.log)
OUT_PP2=$(ask)
sleep 8   # workers write the log asynchronously; without this the slice is empty
# Compare only the taps produced by the REQUEST, and only the numeric payload:
# the line prefix carries the rank name and pid ("Worker_PP3 pid=6801"), which
# differ by construction, so a whole-line diff can never pass at PP>1. Warmup
# taps are excluded too — at PP>1 the receiving rank sees a zero-filled buffer
# for them, which is correct but not comparable to PP=1.
grep "DSPARK_PROBE.*shape" srv_pp2.log | tail -n +$((MARK_PP2 + 1)) |
  sed 's/.*\[DSPARK_PROBE\] //' > probe_pp2.txt
echo "  taps seen: $(grep -c 'DSPARK_PROBE.*shape' srv_pp2.log)"
echo "  middle ranks exercised: $(( PPN > 2 ? PPN - 2 : 0 ))"
pkill -9 -f "[a]pi_server" 2>/dev/null; sleep 5

echo "##### reference: PP=1"
if serve pp1 1; then echo "PP1 boots"; else echo "PP1 FAIL"; exit 1; fi
MARK_PP1=$(grep -c "DSPARK_PROBE.*shape" srv_pp1.log)
OUT_PP1=$(ask)
sleep 8
grep "DSPARK_PROBE.*shape" srv_pp1.log | tail -n +$((MARK_PP1 + 1)) |
  sed 's/.*\[DSPARK_PROBE\] //' > probe_pp1.txt
pkill -9 -f "[a]pi_server" 2>/dev/null

echo "##### GATE 2: tap fingerprints must match"
if [ ! -s probe_pp1.txt ] || [ ! -s probe_pp2.txt ]; then
  # An empty file would make `diff` succeed and report a pass on no evidence.
  echo "GATE2 FAIL - no request taps captured (PP1=$(wc -l < probe_pp1.txt), PP$PPN=$(wc -l < probe_pp2.txt))"
elif diff -q probe_pp1.txt probe_pp2.txt >/dev/null 2>&1; then
  echo "GATE2 PASS - request taps identical at PP=1 and PP=$PPN ($(wc -l < probe_pp1.txt) taps)"
  cat probe_pp1.txt
else
  echo "GATE2 FAIL - taps differ:"; echo "--- PP=1:"; cat probe_pp1.txt; echo "--- PP=$PPN:"; cat probe_pp2.txt
fi

echo "##### GATE 3: output parity + drafts accepted"
[ "$OUT_PP1" = "$OUT_PP2" ] && echo "GATE3a PASS - identical text" || {
  echo "GATE3a FAIL"; echo "  PP1: $OUT_PP1"; echo "  PP2: $OUT_PP2"; }
echo "  acceptance counters (PP=2):"
grep -iE "accept|draft" srv_pp2.log | grep -viE "error|warn" | tail -3 | cut -c1-180
echo DSPARK_PP_TEST_DONE
