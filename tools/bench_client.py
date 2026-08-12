#!/usr/bin/env python3
"""Measure DECODE speed, not whole-pass throughput.

The first version of this harness reported aggregate tokens/s over a pass with
prefix caching on and the first pass discarded. That is biased in favour of
speculation and a reviewer will say so: after pass 1 every prompt is a full
prefix-cache hit, so prefill costs ~nothing in every measured pass. Prefill is
the shared component that speculation does NOT accelerate, so deleting it from
the denominator inflates the relative gain. Discarding the warmup does not fix
that - it removes the one pass that had honest prefill.

So: stream every request, split TTFT from decode, and report

    TPOT = (wall - TTFT) / (output_tokens - 1)

which is what "decode speedup" literally means and which prefix caching cannot
touch.

Second fix: concurrency. Speculation's benefit is largest at batch 1, and under
PP batch 1 is doubly flattering - with one request in flight, pp_size-1 stages
sit idle and speculation partially fills bubbles that real concurrency would
fill anyway. Batch 1 alone is not a defensible claim about a pipeline-parallel
deployment, so we sweep concurrency and report where the crossover sits.

Third: we do NOT print a confidence interval over repeats of a deterministic
greedy workload - that is a CI on timer jitter. Repeats establish stability
("max deviation across reps"); the honest error bar is the spread ACROSS
PROMPTS, since acceptance is strongly content-dependent.

Also hashes the generated text: spec decode is lossless under greedy only up to
numerics, and a shifted EOS would change the token count we divide by.
"""

import argparse
import hashlib
import json
import queue
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a race condition is, with a short example.",
    "Refactor this loop to use a dict comprehension: for k in keys: d[k] = f(k)",
    "Write a SQL query returning the top 5 customers by total order value.",
    "What does the Linux OOM killer do and how do you tune it?",
    "Implement binary search over a rotated sorted array in Go.",
    "Describe the difference between a process and a thread.",
    "Write a bash one-liner that finds the 10 largest files under /var.",
]


def one(port: int, prompt: str, idx: int, max_tokens: int,
        model: str = "m") -> dict:
    """One streamed completion. TTFT is the first chunk carrying text."""
    # include_usage matters specifically for the case being measured: with
    # speculative decoding a single SSE chunk can carry several accepted tokens,
    # so counting chunks undercounts exactly where the speedup is, and would
    # report a spec-on TPOT inflated by the acceptance length. Take the count
    # from the server.
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", body,
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    text = []
    n_chunks = 0
    usage = None
    # Print the server's reason rather than a bare HTTPError traceback: a failed
    # request here has twice cost a whole measurement pass on a model that takes
    # twelve minutes to load.
    try:
        _resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code} from server: {body}") from None
    with _resp as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or [{}]
            piece = choices[0].get("text", "")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(piece)
                n_chunks += 1
    wall = time.perf_counter() - t0
    joined = "".join(text)
    n = (usage or {}).get("completion_tokens") or n_chunks
    decode_s = wall - (ttft or wall)
    return {
        "prompt_idx": idx,
        "ttft_s": ttft,
        "wall_s": wall,
        "decode_s": decode_s,
        "out_tokens": n,
        "n_chunks": n_chunks,
        # Per-token decode cost. n-1 because the first token is TTFT, not decode.
        "tpot_ms": (decode_s / (n - 1) * 1000) if n > 1 else None,
        "text_sha": hashlib.sha256(joined.encode()).hexdigest()[:16],
        "text_head": joined[:80],
    }


def run_pass(port: int, conc: int, max_tokens: int,
             model: str = "m") -> list[dict]:
    """Issue every prompt with `conc` in flight at once."""
    if conc == 1:
        return [one(port, p, i, max_tokens, model) for i, p in enumerate(PROMPTS)]
    work: queue.Queue = queue.Queue()
    for i, p in enumerate(PROMPTS):
        work.put((i, p))
    out: list[dict] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                i, p = work.get_nowait()
            except queue.Empty:
                return
            r = one(port, p, i, max_tokens, model)
            with lock:
                out.append(r)

    ts = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return sorted(out, key=lambda r: r["prompt_idx"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18300)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/workspace/k3/bench/bench2.jsonl")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    # Hardcoding this cost a whole K3 baseline pass: the served name was "k3"
    # and every request came back an HTTP error against "m".
    ap.add_argument("--model-name", default="m")
    args = ap.parse_args()

    run_pass(args.port, args.concurrency, min(64, args.max_tokens),
             args.model_name)  # warm
    for rep in range(1, args.reps + 1):
        rows = run_pass(args.port, args.concurrency, args.max_tokens,
                        args.model_name)
        tpots = [r["tpot_ms"] for r in rows if r["tpot_ms"]]
        rec = {
            "tag": args.tag, "rep": rep, "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "median_tpot_ms": sorted(tpots)[len(tpots) // 2] if tpots else None,
            "per_request": rows,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # A dead engine returns zero-token responses; say so and keep going
        # rather than dying on the format string and losing the later configs.
        med = rec["median_tpot_ms"]
        print(f"{args.tag} c={args.concurrency} rep{rep}: "
              + (f"median TPOT {med:.2f} ms" if med is not None
                 else "NO USABLE REQUESTS (all returned <2 tokens)")
              + f" ({len(tpots)}/{len(rows)} requests)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
