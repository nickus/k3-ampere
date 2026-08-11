#!/bin/bash
# Box7 gate: PP=2 fp8_ds_mla + KV-offload validation (task #27). Idempotent-ish.
set -e
cd /workspace/k3
PY=/venv/main/bin/python
PIP=/venv/main/bin/pip
SP=$($PY -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
echo "== [1/7] wheel + patches"
$PY -c "import zipfile; assert zipfile.ZipFile('/workspace/k3/vllm-0.26.1rc5+sm86-cp312-cp312-linux_x86_64.whl').testzip() is None"
$PIP install -q --force-reinstall --no-deps /workspace/k3/vllm-0.26.1rc5+sm86-cp312-cp312-linux_x86_64.whl
$PIP install -q --extra-index-url https://flashinfer.ai/whl/ "flashinfer-cubin==0.6.15.post1" 2>&1 | tail -1 || true
tar xzf vllm_py_head.tgz -C $SP && find $SP/vllm -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
cd $SP && patch -p1 --forward < /workspace/k3/pr48918_novtest.diff >/dev/null 2>&1 || true
$PY /workspace/k3/hand_patches.py $SP | tail -1
cd /workspace/k3 && tar xzf fp8kv_pkg.tgz && cp -r fp8kv-k3-pkg/fp8kv_k3 $SP/
$PY /workspace/k3/fp8kv-k3-pkg/fp8kv_k3/apply_vllm_patches.py $SP | tail -1
echo "== [2/7] slice"
FLASHINFER_DISABLE_VERSION_CHECK=1 $PY /workspace/k3/gen_slice_hf.py | tail -1
$PY - <<'PYEOF'
import json
p = "/workspace/k3/k3-slice-hf/config.json"
c = json.load(open(p))
c["architectures"] = ["KimiLinearForCausalLM"]; c["model_type"] = "kimi_linear"
tc = c.pop("text_config", {})
for k, v in tc.items(): c.setdefault(k, v)
for k in ("vision_config","image_placeholder","media_placeholder_token_id"): c.pop(k, None)
c["auto_map"] = {"AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
                 "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM"}
json.dump(c, open(p, "w"), indent=1)
print("text-only ok")
PYEOF
echo "== [3/7] PP=2 fp8_ds_mla boot+parity"
cat > serve27.sh <<'EOF'
#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_DISABLE_VERSION_CHECK=1
KV=$1; PORT=$2; PP=$3; shift 3
exec /venv/main/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code \
  --load-format dummy --kv-cache-dtype $KV \
  --pipeline-parallel-size $PP --tensor-parallel-size 1 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager --port $PORT "$@"
EOF
chmod +x serve27.sh
kill_all() { for p in $(pgrep -f "api_server|VLLM::"); do kill -9 $p 2>/dev/null; done; sleep 5; }
wait_up() { for i in $(seq 1 90); do curl -s http://127.0.0.1:$1/v1/models 2>/dev/null | grep -q '"id"' && return 0; sleep 5; done; return 1; }
P="In the beginning there was code and the code was"
gen() { curl -s http://127.0.0.1:$1/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"k3\",\"prompt\":\"$P\",\"max_tokens\":32,\"temperature\":0}" | /venv/main/bin/python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null; }
kill_all
(nohup bash serve27.sh fp8_ds_mla 18000 2 > s_pp2_fp8.log 2>&1 &)
wait_up 18000 || { echo "PP2_FP8_BOOT_FAIL"; grep -m2 -iE "error" s_pp2_fp8.log | tail -2; exit 1; }
PP2=$(gen 18000); echo "PP2_FP8_TEXT:::$PP2"
kill_all
(nohup bash serve27.sh fp8_ds_mla 18000 1 > s_pp1_fp8.log 2>&1 &)
wait_up 18000 || { echo "PP1_BOOT_FAIL"; exit 1; }
PP1=$(gen 18000); echo "PP1_FP8_TEXT:::$PP1"
[ "$PP1" = "$PP2" ] && echo "PP_PARITY: EXACT" || echo "PP_PARITY: DIFFER"
kill_all
echo "GATE_PP2_DONE"
