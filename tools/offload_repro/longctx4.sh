#!/bin/bash
cd /workspace/k3
for p in $(pgrep -f "api_server|VLLM::"); do kill -9 $p 2>/dev/null; done; sleep 5
rm -rf kv_nvme; mkdir -p kv_nvme
(nohup bash srv5.sh > srv5.log 2>&1 &)
for i in $(seq 1 60); do curl -s http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q "\"id\"" && break; sleep 10; done
curl -s http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q "\"id\"" || { echo LONG_BOOT_FAIL; exit 1; }
echo LONG_BOOT_OK
/venv/main/bin/python - <<"PY"
import json, time, urllib.request, subprocess, random, os

U = "http://127.0.0.1:18000/v1/completions"

def call(p, lp=False):
    body = {"model": "k3", "prompt": p, "max_tokens": 1, "temperature": 0}
    if lp: body["logprobs"] = 1
    t = time.time()
    r = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    dt = time.time() - t
    tl = d["choices"][0].get("logprobs", {}).get("token_logprobs") if lp else None
    return d.get("usage", {}).get("prompt_tokens", 0), dt, tl

def mk(nw, seed):
    rnd = random.Random(seed)           # independent stream -> no shared prefix
    return " ".join(f"tok{rnd.randint(0,9999)}" for _ in range(nw))

def files():
    out = []
    for root, _, fs in os.walk("/workspace/k3/kv_nvme"):
        for f in fs: out.append(os.path.join(root, f))
    return out

def nvme_mb():
    t = 0
    for f in files():
        try: t += os.path.getsize(f)
        except OSError: pass
    return t / 1e6

def evict_page_cache():
    """POSIX_FADV_DONTNEED drops these files' pages without root."""
    n = 0
    for f in files():
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
            finally: os.close(fd)
        except OSError: pass
    return n

def read_bytes():
    tot = 0
    pids = subprocess.run("pgrep -f 'api_server|VLLM::'", shell=True, capture_output=True, text=True).stdout.split()
    for p in pids:
        try:
            for ln in open(f"/proc/{p}/io"):
                if ln.startswith("read_bytes:"): tot += int(ln.split()[1])
        except OSError: pass
    return tot

call(mk(200, 7))  # warm up server / graphs so row 1 isn't polluted
print(" words   ptok  cold_s  nvme_s   gpu_s  cold_t/s  nvme_t/s   gpu_t/s  spd_nvme  diskMB  MB/s  maxdlogp", flush=True)
for words in (1400, 2800, 5600, 9800):
    P = mk(words, 1000 + words)                       # unique stream per size
    ptok, t_cold, lp_cold = call(P, lp=True)
    for j in range(5):                                # 5 x this size >> 81920-token GPU cache
        call(mk(words, 50000 + words * 10 + j))
    evict_page_cache()
    rb0 = read_bytes()
    _, t_nvme, lp_warm = call(P, lp=True)             # must come back from CPU/NVMe tier
    disk = (read_bytes() - rb0) / 1e6
    _, t_gpu, _ = call(P)                             # now resident in GPU cache: upper bound
    dl = max(abs((a or 0) - (b or 0)) for a, b in zip(lp_cold, lp_warm)) if (lp_cold and lp_warm) else -1
    print(f"{words:>6} {ptok:>6} {t_cold:>7.2f} {t_nvme:>7.2f} {t_gpu:>7.2f} "
          f"{ptok/max(t_cold,1e-9):>9.0f} {ptok/max(t_nvme,1e-9):>9.0f} {ptok/max(t_gpu,1e-9):>9.0f} "
          f"{t_cold/max(t_nvme,1e-9):>8.2f}x {disk:>7.1f} {disk/max(t_nvme,1e-9):>5.0f} {dl:>9.6f}", flush=True)
print(f"NVMe tier total: {nvme_mb():.0f} MB in {len(files())} files", flush=True)
PY
for p in $(pgrep -f "api_server|VLLM::"); do kill -9 $p 2>/dev/null; done
echo LONG_DONE
