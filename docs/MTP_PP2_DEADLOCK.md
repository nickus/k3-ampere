# K3 MTP + PP=2: rank-asymmetric init deadlock (diagnosed 2026-07-30)

Status: root-caused to phase & mechanism; exact frame needs SYS_PTRACE
(vast container forbids py-spy/gdb). Upstream issue draft below — **not
filed yet** (outbound requires approval).

## Symptom

`vllm serve <K3> --pipeline-parallel-size 2 --spec-method kimi_k3_mtp
--spec-tokens 1` (V2 runner): Worker_PP0 dies **silently** (exit code None,
no python traceback) at **~11 min** after start. PP1 then dies with gloo
"Connection closed by peer" inside a normal RPC `recv_object`. PP=1 same
config: boots and drafts (23/23 drafts counted).

## Evidence chain (all runs on 2×3090, wheel 0.26.1rc2+sm86 @38a267cdd)

1. Death at ~11 min in 2/2 default-config runs; **disabling
   `enable_jit_warmup` does NOT help** (same silent death, same phase) —
   the spec-KDA warmup is innocent.
2. With `--distributed-timeout-seconds 3600 --cpu-distributed-timeout-seconds
   3600` the death **disappears** — replaced by an indefinite hang (25+ min
   observed): PP0 main thread spinning 100% CPU + GPU0 100% util, PP1
   parked on futex, Triton cache NOT growing (722→722 entries/45 s — not
   compilation), EngineCore printing "No available shared memory broadcast
   block" forever. ⇒ the ~11 min was the **default ~600 s dist-watchdog
   killing a deadlocked rank** (silent native abort), not a kernel crash.
3. Phase: PP0's last log lines = memory-profiling KV report, then the
   `parallel_state.py:834` `frombuffer` warning — i.e. inside a CPU-group
   `send_object` during **post-profiling initialization** (KV-cache /
   drafter init under V2+PP). The GPU spin is consistent with an enqueued
   NCCL collective whose peer never arrives.
4. Asymmetry: only the MTP config deadlocks; Phase A/B (no spec) at PP=2
   boot fine through the same init. Drafter exists only on the last rank ⇒
   a collective reached on one rank but not the other in the V2 spec-decode
   init path.

## Workarounds for the rig (none great)

- MTP+PP currently unusable for K3 → run PP without spec decode (base K3
  ships no MTP head anyway, so nothing is lost **today**).
- Raising dist timeouts converts crash→hang; do NOT use as a workaround.

## Upstream issue draft (file after approval)

Title: `[Bug] Kimi-K3 MTP + PP>1 (V2 runner): rank-asymmetric collective in
post-profiling init deadlocks; watchdog kills rank 0 after dist timeout`
Body: symptom, evidence chain 1–4 above, repro = synthetic
KimiLinearForCausalLM slice (config attached) + `--load-format dummy` +
flags; note PP=1 healthy, warmup not involved, ptrace unavailable in the
repro environment. Suggest auditing V2 `SpecDecode`/KV-init collectives for
`is_last_rank`-conditional paths.
