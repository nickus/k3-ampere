#!/bin/bash
# Bring a fresh vast.ai box to the exact state the DSpark-under-PP gates need.
# Idempotent: safe to re-run after a partial failure.
#
# Assumes: tools/ already uploaded to /workspace/k3, driver supports CUDA 13
# (check cuda_max_good >= 12.8 BEFORE renting — a driver-535 box reports
# healthy and then cannot run a single cu130 wheel).
set -eu
WORK=/workspace/k3
VENV=/venv/nm
mkdir -p "$WORK"
cd "$WORK"

echo "=== [1/6] system deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq python3-venv python3-pip git curl >/dev/null 2>&1 || true

echo "=== [1b/6] measure REAL egress before committing 20 minutes to a download"
# vast.ai's advertised inet_down is a speedtest figure and says nothing about
# throughput to the registries we actually use. One rented host advertised
# 1412 Mbit/s and delivered 901 B/s from wheels.vllm.ai and 0 B/s from PyPI -
# pip crawled for 20 minutes before that became obvious. 20 seconds here.
SPEED=$(curl -s -o /dev/null -w '%{speed_download}' --max-time 20 \
  https://wheels.vllm.ai/nightly/cu130/vllm/ || echo 0)
SPEED=${SPEED%%.*}
echo "egress to wheels.vllm.ai: ${SPEED} B/s"
if [ "${SPEED:-0}" -lt 50000 ]; then
  echo "ABORT: this host cannot reach the wheel index at a usable speed."
  echo "Destroy it and rent another - do not wait this out."
  exit 2
fi

echo "=== [2/6] venv + vLLM nightly (cu130)"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
NIGHTLY=https://wheels.vllm.ai/nightly/cu130
if ! "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
  # Getting this right took three attempts, so the reasoning stays here:
  #   * `pip install vllm --pre`            -> resolves to the RELEASE (twice bitten)
  #   * `--index-url $NIGHTLY` alone        -> vLLM's own deps (regex, ...) 404,
  #                                            the nightly index hosts only vLLM
  #   * `--extra-index-url $NIGHTLY`        -> PyPI wins again, because the release
  #                                            version sorts above 0.26.1rc1.devNNN
  # So: pin the wheel by its exact URL (version can no longer be renegotiated)
  # and let the dependencies resolve from PyPI as usual. The href in the index
  # is relative and points at a per-commit directory, so resolve it rather than
  # constructing the URL by hand — a hand-built path 404s.
  REL=$(curl -s "$NIGHTLY/vllm/" | grep -oE 'href="[^"]+x86_64\.whl"' | tail -1 |
        sed 's/href="//; s/"//')
  [ -n "$REL" ] || { echo "cannot find a nightly wheel at $NIGHTLY"; exit 1; }
  URL=$(python3 -c "import urllib.parse;print(urllib.parse.urljoin('$NIGHTLY/vllm/','$REL'))")
  echo "wheel: $URL"
  "$VENV/bin/pip" install -q --extra-index-url "$NIGHTLY" "$URL"
fi
"$VENV/bin/pip" install -q py-spy >/dev/null 2>&1 || true

echo "=== [3/6] assert we got main, not a release"
"$VENV/bin/python" - <<'PY'
import sys, vllm, torch
v = vllm.__version__
print("vllm:", v, "| torch:", torch.__version__, "| cuda:", torch.version.cuda)
assert ".dev" in v, f"NOT a nightly wheel: {v} — the whole point was to test main"
assert torch.cuda.is_available(), "no CUDA visible"
print("gpus:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
assert cap == (8, 6), f"expected sm_86, got {cap}"
PY

echo "=== [4/6] build the slice + draft"
[ -d "$WORK/k3-slice-hf" ]    || "$VENV/bin/python" "$WORK/gen_slice_hf.py"
[ -d "$WORK/k3-dspark-draft" ] || "$VENV/bin/python" "$WORK/gen_dspark_draft.py"
ls -d "$WORK"/k3-slice-hf "$WORK"/k3-dspark-draft

echo "=== [5/6] apply patches to the installed tree"
"$VENV/bin/python" "$WORK/dspark_pp_patch.py"
"$VENV/bin/python" "$WORK/slice_eagle3_shim.py"

echo "=== [6/6] import check (a circular import here kills every worker later)"
"$VENV/bin/python" -c "
import vllm.v1.worker.gpu.model_runner as m
import vllm.v1.worker.gpu.warmup as w
import vllm.v1.attention.backends.mla.triton_mla as t
print('imports OK')
"
echo '=== BOOTSTRAP_DONE'
