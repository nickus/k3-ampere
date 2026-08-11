#!/bin/bash
cd /workspace/k3
for p in $(pgrep -f "api_server|VLLM::"); do kill -9 $p 2>/dev/null; done; sleep 5
rm -rf kv_nvme; mkdir -p kv_nvme
cat > srv5.sh <<"EOF"
#!/bin/bash
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_DISABLE_VERSION_CHECK=1
exec /venv/main/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/k3/k3-slice-hf --served-model-name k3 --trust-remote-code \
  --load-format dummy --kv-cache-dtype auto \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --enable-prefix-caching --block-size 512 --num-gpu-blocks-override 160 \
  --max-model-len 32768 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.82 --enforce-eager --port 18000 \
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":268435456,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"/workspace/k3/kv_nvme\"}]}}"
EOF
chmod +x srv5.sh; (nohup bash srv5.sh > srv5.log 2>&1 &)
for i in $(seq 1 60); do curl -s http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q "\"id\"" && break; sleep 10; done
curl -s http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q "\"id\"" || { echo LONG_BOOT_FAIL; grep -viE "INFO|WARNING" srv5.log | grep -iE "error|assert" | tail -3 | cut -c1-170; exit 1; }
echo LONG_BOOT_OK
/venv/main/bin/python - <<"PY"
import json, time, urllib.request, subprocess, random, os, glob
U = "http://127.0.0.1:18000/v1/completions"

def call(p, mt=1, lp=False):
    body = {"model": "k3", "prompt": p, "max_tokens": mt, "temperature": 0}
    if lp:
        body["logprobs"] = 1
    t = time.time()
    r = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    dt = time.time() - t
    ch = d["choices"][0]
    tl = ch.get("logprobs", {}).get("token_logprobs") if lp else None
    return d.get("usage", {}).get("prompt_tokens", 0), dt, tl

def mk(n, seed):
    random.seed(seed)
    return " ".join(f"tok{random.randint(0,9999)}" for _ in range(n))

def nvme():
    tot = 0; cnt = 0
    for root, _, fs in os.walk("/workspace/k3/kv_nvme"):
        for f in fs:
            try: tot += os.path.getsize(os.path.join(root, f)); cnt += 1
            except OSError: pass
    return tot / 1e6, cnt

def srv_pids():
    out = subprocess.run("pgrep -f 'api_server|VLLM::'", shell=True, capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit()]

def read_bytes():
    tot = 0
    for p in srv_pids():
        try:
            for ln in open(f"/proc/{p}/io"):
                if ln.startswith("read_bytes:"): tot += int(ln.split()[1])
        except OSError: pass
    return tot

def drop_caches():
    try:
        subprocess.run("sync", shell=True, timeout=60)
        with open("/proc/sys/vm/drop_caches", "w") as f: f.write("3")
        return True
    except Exception:
        return False

print("  words   ptok  cold_s  rest_s  speedup  prefill_t/s  restore_t/s  nvme_MB  files  diskMB  cache  maxdlogp")
for words in (2000, 6000, 12000, 20000):
    P = mk(words, 42)
    ptok, tc, lp_cold = call(P, lp=True)
    # evict from GPU: 160 blocks x 512 = 81920 tokens of GPU KV
    for j in range(5):
        call(mk(words, 100 + j))
    dropped = drop_caches()
    rb0 = read_bytes()
    mb, files = nvme()
    ptok2, tr, lp_warm = call(P, lp=True)
    diskmb = (read_bytes() - rb0) / 1e6
    dl = 0.0
    if lp_cold and lp_warm:
        dl = max(abs((a or 0) - (b or 0)) for a, b in zip(lp_cold, lp_warm))
    print(f"{words:>7} {ptok:>6} {tc:>7.2f} {tr:>7.2f} {tc/max(tr,1e-9):>7.2f}x {ptok/max(tc,1e-9):>12.0f} {ptok/max(tr,1e-9):>12.0f} {mb:>8.0f} {files:>6} {diskmb:>7.1f} {'cold' if dropped else 'warm':>6} {dl:>9.6f}", flush=True)
PY
echo "=== offload events:"; grep -ciE "offload|evict" srv5.log 2>/dev/null
for p in $(pgrep -f "api_server|VLLM::"); do kill -9 $p 2>/dev/null; done
echo LONG_DONE
