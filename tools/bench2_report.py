#!/usr/bin/env python3
"""Turn bench2.jsonl into the numbers we are willing to defend.

Deliberately does NOT print a confidence interval. Five repeats of a
deterministic greedy workload against one warm server process measure timer
jitter, not the variance anyone cares about; a +/- on that is fake precision.
Repeats are used for exactly one claim - that the measurement is stable - and
the honest error bar is the spread ACROSS PROMPTS, because acceptance rate is
strongly content-dependent and eight prompts is a small sample of content.

Speedup is computed PER PROMPT (same prompt, same concurrency, spec on vs off)
and aggregated with the geometric mean, which is the correct average for ratios.
"""

import argparse
import collections
import json
import math


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="/workspace/k3/bench/bench2.jsonl")
    args = ap.parse_args()

    # (pp, spec, concurrency, prompt) -> [tpot across reps]
    cell: dict[tuple, list[float]] = collections.defaultdict(list)
    texts: dict[tuple, set] = collections.defaultdict(set)
    for line in open(args.file):
        r = json.loads(line)
        pp, spec = r["tag"].split("_spec")
        for req in r["per_request"]:
            if req["tpot_ms"] is None:
                continue
            key = (pp, spec, r["concurrency"], req["prompt_idx"])
            cell[key].append(req["tpot_ms"])
            texts[(pp, r["concurrency"], req["prompt_idx"], spec)].add(
                req["text_sha"]
            )

    # Stability: how far apart are repeats of the same cell?
    devs = [
        (max(v) - min(v)) / (sum(v) / len(v))
        for v in cell.values() if len(v) > 1
    ]
    if devs:
        print(f"rep stability: max deviation across repeats = {max(devs) * 100:.2f}%"
              f" (n={len(devs)} cells)")

    print()
    print(f"{'PP':>3} {'conc':>5} {'TPOT off':>9} {'TPOT on':>9} "
          f"{'speedup':>8}  {'per-prompt range':>18}")
    for pp in sorted({k[0] for k in cell}):
        for c in sorted({k[2] for k in cell if k[0] == pp}):
            ratios, offs, ons = [], [], []
            for p in sorted({k[3] for k in cell if k[0] == pp and k[2] == c}):
                off = cell.get((pp, "no", c, p))
                on = cell.get((pp, "yes", c, p))
                if not off or not on:
                    continue
                mo, mn = sum(off) / len(off), sum(on) / len(on)
                offs.append(mo)
                ons.append(mn)
                ratios.append(mo / mn)
            if not ratios:
                continue
            print(f"{pp:>3} {c:>5} {sum(offs) / len(offs):>8.2f}m "
                  f"{sum(ons) / len(ons):>8.2f}m {geomean(ratios):>7.3f}x  "
                  f"{min(ratios):>7.3f}-{max(ratios):.3f}")

    # Greedy parity: spec decode is lossless up to numerics, but a shifted EOS
    # changes the token count we divide by, so a mismatch invalidates the ratio.
    print()
    bad = 0
    for pp in sorted({k[0] for k in texts}):
        for c in sorted({k[1] for k in texts if k[0] == pp}):
            for p in sorted({k[2] for k in texts if k[0] == pp and k[1] == c}):
                a = texts.get((pp, c, p, "no"), set())
                b = texts.get((pp, c, p, "yes"), set())
                if a and b and a != b:
                    bad += 1
                    print(f"  PARITY MISMATCH pp={pp} c={c} prompt={p}: "
                          f"off={sorted(a)} on={sorted(b)}")
    print(f"greedy parity: {'OK' if bad == 0 else f'{bad} MISMATCHES'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
