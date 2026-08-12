#!/bin/bash
# Two questions in one run, both discriminating, neither needing new machinery.
#
# Q1. Is the divergence tied to a REJECTION? The standing hypothesis is that
#     non-last ranks advance num_computed_tokens optimistically by the full query
#     length and only receive the `-num_rejected` correction pp_size steps later
#     (see _post_update_kernel: query_start_loc is None on those ranks, so
#     computed_delta = -num_rejected). If so, nothing can go wrong until a draft
#     is actually rejected — without speculation num_rejected is always 0, which
#     is exactly why PP alone is clean.
#
# Q2. Is the divergence at an ABSOLUTE position or at the END of generation?
#     Run the same prompt at max_tokens 24 and 64. If it tracks the end, the
#     finishing path is implicated instead.
set -u
cd /workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0 SKIP_KERNEL_WARMUP=1 PYTHONPATH=/workspace/k3
PORT=18050

cat > sitecustomize.py <<'EOF'
try:
    import dspark_pp_probe  # noqa: F401
except Exception as e:
    print("[DSPARK_PROBE] not installed:", e, flush=True)
EOF

serve() {  # serve <pp> <spec yes|no> <logfile>
  pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 6
  local args=(--model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code
    --load-format dummy --pipeline-parallel-size "$1" --tensor-parallel-size 1
    --enable-prefix-caching --block-size 512 --max-model-len 4096
    --gpu-memory-utilization 0.82 --enforce-eager --port $PORT)
  [ "$2" = yes ] && args+=(--speculative-config '{"model":"/workspace/k3/k3-dspark-draft","method":"dspark","num_speculative_tokens":3}')
  nohup /venv/nm/bin/python -m vllm.entrypoints.openai.api_server "${args[@]}" > "$3" 2>&1 &
  for i in $(seq 1 45); do
    sleep 10
    curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && return 0
  done
  return 1
}

ids() {  # ids <max_tokens>
  curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"k3\",\"prompt\":\"The quick brown fox jumps over the lazy dog and then\",\"max_tokens\":$1,\"temperature\":0,\"logprobs\":1}" \
    | PYTHONPATH= /venv/nm/bin/python -c "
import sys,json
d=json.load(sys.stdin)['choices'][0]
print(' '.join((d.get('logprobs') or {}).get('tokens') or []))" 2>/dev/null
}

echo "########## Q2: absolute position or end of generation?"
for N in 24 64; do
  serve 1 no  "q2_ref_$N.log"  || { echo "no serve"; exit 1; }; R=$(ids $N)
  serve 2 yes "q2_pp2_$N.log"  || { echo "no serve"; exit 1; }; S=$(ids $N)
  PYTHONPATH= /venv/nm/bin/python - "$R" "$S" "$N" <<'PY'
import sys
a=sys.argv[1].split(); b=sys.argv[2].split(); n=int(sys.argv[3])
if not a or not b:
    print(f"max_tokens={n}: EMPTY (ref={len(a)} pp2={len(b)})"); raise SystemExit
for i,(x,y) in enumerate(zip(a,b)):
    if x!=y:
        print(f"max_tokens={n}: first divergence at #{i} of {len(a)} -> {len(a)-i} tokens from the end")
        break
else:
    print(f"max_tokens={n}: identical over {min(len(a),len(b))} tokens")
PY
done

echo "########## Q1: does the divergence coincide with the first rejection?"
/venv/nm/bin/python verify_probe.py 2>&1 | tail -1
serve 1 yes "q1_pp1.log" || exit 1; P1=$(ids 24)
serve 2 yes "q1_pp2.log" || exit 1; P2=$(ids 24)
for f in q1_pp1 q1_pp2; do
  grep "VERIFY" "$f.log" | sed 's/.*\[VERIFY\] //; s/pid=[0-9]* //' > "$f.txt"
done
PYTHONPATH= /venv/nm/bin/python - "$P1" "$P2" <<'PY'
import re, sys
a=sys.argv[1].split(); b=sys.argv[2].split()
div=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None)
print("token divergence at:", div, "of", len(a))
for tag in ("q1_pp1","q1_pp2"):
    rej=[]
    for n,line in enumerate(open(f"/workspace/k3/{tag}.txt")):
        m=re.search(r"rejected=\[([^\]]*)\]", line)
        if m and any(v.strip() not in ("0","") for v in m.group(1).split(",")):
            rej.append(n)
    print(f"{tag}: steps with a non-zero rejection: {rej[:10]} (total {len(rej)})")
PY
pkill -9 -f "[a]pi_server" 2>/dev/null; pkill -9 -f "[V]LLM::" 2>/dev/null
echo PARITY_HUNT_DONE
