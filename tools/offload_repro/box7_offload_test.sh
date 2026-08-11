#!/bin/bash
# Part 2 of task #27: KV-offload connectors on PP=2 hybrid slice.
# Run AFTER box7_offload_gate.sh (stack installed, slice ready).
set -e
cd /workspace/k3
PY=/venv/main/bin/python
kill_all() { for p in $(pgrep -f "api_server|VLLM::"); do kill -9 $p 2>/dev/null; done; sleep 5; }
wait_up() { for i in $(seq 1 90); do curl -s http://127.0.0.1:$1/v1/models 2>/dev/null | grep -q '"id"' && return 0; sleep 5; done; return 1; }
LONGP=$($PY -c "print('def f_%d(): pass' * 300 % tuple(range(300)))")
gen() { curl -s http://127.0.0.1:$1/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"k3\",\"prompt\":\"$LONGP\",\"max_tokens\":24,\"temperature\":0}" | $PY -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null; }

echo "== [A] NATIVE OffloadingConnector, PP=2, bf16 KV (control: known-bug hunt #50821/#46453)"
kill_all
(nohup bash serve27.sh auto 18000 2 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":4096}}' \
  > s_native_off.log 2>&1 &)
if wait_up 18000; then
  A1=$(gen 18000); echo "NATIVE_GEN1:::${A1:0:60}"
  A2=$(gen 18000); echo "NATIVE_GEN2(cache-hit):::${A2:0:60}"
  [ "$A1" = "$A2" ] && echo "NATIVE_RESUME_PARITY: EXACT" || echo "NATIVE_RESUME_PARITY: DIFFER  <-- hybrid state bug?"
else
  echo "NATIVE_BOOT: FAIL"
  grep -m3 -iE "error|assert|mamba|hybrid" s_native_off.log | tail -3 | cut -c1-200
fi
kill_all

echo "== [B] NATIVE + fp8_ds_mla pages (656B custom page-size)"
(nohup bash serve27.sh fp8_ds_mla 18000 2 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":4096}}' \
  > s_native_fp8.log 2>&1 &)
if wait_up 18000; then
  B1=$(gen 18000); B2=$(gen 18000)
  [ "$B1" = "$B2" ] && echo "NATIVE_FP8_RESUME: EXACT" || echo "NATIVE_FP8_RESUME: DIFFER"
else
  echo "NATIVE_FP8_BOOT: FAIL"
  grep -m3 -iE "error|assert|page|656" s_native_fp8.log | tail -3 | cut -c1-200
fi
kill_all

echo "== [C] LMCache (documented K3-hybrid support), PP=2"
/venv/main/bin/pip install -q lmcache 2>&1 | tail -1 || echo "lmcache install failed"
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_DISK="file:///workspace/k3/lmcache_disk/" LMCACHE_MAX_LOCAL_DISK_SIZE=20
mkdir -p /workspace/k3/lmcache_disk
(nohup bash serve27.sh auto 18000 2 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  > s_lmcache.log 2>&1 &)
if wait_up 18000; then
  C1=$(gen 18000); echo "LMCACHE_GEN1:::${C1:0:60}"
  C2=$(gen 18000)
  [ "$C1" = "$C2" ] && echo "LMCACHE_RESUME: EXACT" || echo "LMCACHE_RESUME: DIFFER"
  echo "disk objects: $(find /workspace/k3/lmcache_disk -type f 2>/dev/null | wc -l)"
else
  echo "LMCACHE_BOOT: FAIL"
  grep -m3 -iE "error|assert" s_lmcache.log | tail -3 | cut -c1-200
fi
kill_all
echo "OFFLOAD_TESTS_DONE"
