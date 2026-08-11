# Twenty things I would tell myself before starting this again

Written 2026-08-11, after bringing up Kimi-K3 on Ampere, proving KV offload, and
getting DSpark half-working under pipeline parallelism. Every item cost
something real — time, money, or a retracted claim. Ordered by how much.

## About evidence

**1. Verify a finding against fresh upstream immediately before you report it.**
I filed three "upstream bugs" and two evaporated on re-check: one had been fixed
on main, one had been documented. The build I was working from was 11 days old,
which in vLLM is ~50 commits. Cite the SHA in the report.

**2. "Fixed upstream" and "fixed in the release people install" are different
claims.** The `index_fill_` crash was fixed on main and still shipped in 0.27.0,
published that same week. If you say "already fixed", say *where*.

**3. An error message can be stronger evidence than a passing test.**
`mat1 and mat2 shapes cannot be multiplied (2048x3072 and 4096x1024)` proved the
tap plumbing worked: the draft received 3 hidden-state taps where 4 were
expected, and one of those 3 was computed on a different GPU. A green check
could have come from a dozen accidents; that dimension could not.

**4. Before fixing something, prove it is actually broken the way you think.**
The baseline step ("confirm upstream refuses this before patching") is what
revealed the real guard was two levels above the one I had read. Without it I
would have "fixed" a path execution never reached.

**5. For silent failure modes, booting proves nothing — compare values.**
Dropping 4 of 5 hidden-state taps does not raise; the model just gets worse.
The only meaningful gate was a position-weighted fingerprint of the taps at
PP=1 versus PP=2, with the weighting chosen so a *reordered* set is
distinguishable from a correct one.

**6. Ask the system what it requires instead of guessing.**
Two failed attempts at satisfying `SupportsEagle3` ended the moment I printed
`SupportsEagle3.__protocol_attrs__` and diffed it against the class. Runtime
protocols check member *presence*; the missing pieces were two flags, not the
methods I kept rewriting.

**7. Keep your retractions in the record.**
I published "K3-Q2 loses to GLM-5.2" (one tester, one task, a homemade quant)
and "KV offload is architecturally unready for K3" (actually three config
requirements and one bug). Both are still in the journal, marked wrong. A
document that only contains correct conclusions teaches nothing about how
confident to be next time.

## About estimating work

**8. When a task looks like "delete one guard", budget for a ladder.**
DSpark-under-PP was 13 obstacles in series, each visible only after the previous
was removed. Six were upstream; **seven were mine** — five in the test harness,
two in my own patch. The fix was never the expensive part.

**9. The expensive part is a faithful environment, not the change itself.**
Most of that day went into making a synthetic slice behave enough like the real
model to exercise the patch at all.

**10. Scale test fixtures, do not copy them.**
I gave a 1024-wide miniature draft the production `q_lora_rank 1536` — a ratio
that exists in no real checkpoint — and got an illegal memory access inside
Triton. Dimensions that are layout-critical stay; dimensions that are
proportional must be scaled.

**11. Separate the upstream patch from the harness hack, physically.**
`slice_eagle3_shim.py` lives in its own file with "TEST ONLY, do not propose
upstream" at the top. Mixing them would have poisoned a contribution.

## About distributed systems

**12. A collective must be symmetric — check the call path actually executes on
every rank.** I "fixed" an asymmetric broadcast by moving it later in the same
function, which changed nothing: `load_dspark_model` runs only on the last rank,
because `init_speculator` sits under `if self.is_last_pp_rank`. No reordering
inside a function that one rank never enters can help.

**13. Bugs that need PP≥2 to appear are invisible in every single-GPU test.**
The draft offsets its layer names by the layer count *on this rank*; at PP=1
that happens to equal the total, so the collision only exists in a pipeline.
Expect a whole class of these and look for the quantity that "coincidentally"
matches at PP=1.

**14. Blast radius can scale with the deployment, not with the repro.**
The block-size alignment bug touches 2 of 25 ranks at PP=25 and 26 of 50 at
PP=50 — at rig scale it is the majority case, not a corner case. Compute that
number; it changes an issue's priority more than any prose.

**15. Distinguish capacity from speed when advising on hardware.**
More cards do not add KV cache: vLLM asserts every rank has the same block
count, so the tightest card sets it, and that card still holds two layers. More
cache comes from a smaller checkpoint. Extra cards buy redundancy — a real
reason, just not that one.

## About operating rented infrastructure

**16. Check `vllm.__version__` after installing, never trust the command.**
`--extra-index-url pypi` silently made pip prefer a release over main, and I
spent several runs patching the wrong tree. I had written this lesson down that
morning and repeated it that afternoon, because the fix was in my notes and not
in the script. **Put the lesson in the code, not the notes.**

**17. Never edit an installed tree by crude text deletion.**
Removing "my old patch" with a `find`/slice one-liner also removed the class's
`__init__`, and I spent twenty minutes debugging `object has no attribute
'model'` — a symptom I had created. Restore from the wheel and re-apply.

**18. `pkill -f "foo.sh"` over SSH matches the SSH command's own command line.**
It kills the launcher before it launches anything, and the failure looks like a
silent no-op. Bracket the first character (`[f]oo.sh`) or put the launch in a
script file so the pattern never appears in the invoking command.

**19. Pull the whole evidence bundle before destroying a box.**
I once saved a single 570-byte output file and lost the raw logs behind a
published results table. The numbers were reported accurately and could no
longer be re-derived from an artifact. Tar the directory.

**20. Rented boxes are disposable; your time is not.**
Stopped instances are not durable (one was evicted with its 1 TB volume after
10 idle days). Offer IDs are ephemeral — only `machine_id` is stable, so record
it for a host that worked. And when a box misbehaves, destroy it within minutes
rather than nursing it: eight failed boxes in one afternoon cost cents in rent
and hours in attention.

---

**The one that generalises past this project:** when work is queued on something
slow, do the part that does not need it. The hour I spent waiting on background
jobs produced nothing, and the same hour spent reading code found the defect
that the whole patch turned out to be about.
