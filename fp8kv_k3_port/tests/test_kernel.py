# SPDX-License-Identifier: Apache-2.0
"""Patch 4: ds_mla decode kernel vs an fp32 naive reference.

CPU-runnable (TRITON_INTERPRET=1). Queries are fp32: the interpreter cannot
do bf16 arithmetic (stores bf16 as raw uint16) — the bf16 production path is
covered by the GPU-gated parity test on the box.
"""

import math

import pytest
import torch

from fp8kv_k3.kernel import decode_attention_fwd_ds_mla
from fp8kv_k3.layout import NOPE, ROPE, dequantize_row_ds_mla, quantize_row_ds_mla

from .conftest import DEVICE

torch.manual_seed(11)

BATCH, HEADS, PAGE = 3, 8, 16
SEQ_LENS = [40, 17, 64]


def _naive_ref(q, rows, req_to_token, seq_lens, sm_scale):
    """fp32 attention over golden-dequantized rows."""
    outs, lses = [], []
    for b, s in enumerate(seq_lens):
        toks = []
        for i in range(s):
            page = req_to_token[b, i // PAGE].item()
            toks.append(page * PAGE + i % PAGE)
        kv = dequantize_row_ds_mla(rows[toks])            # [s, 576]
        k, v = kv, kv[:, :NOPE]
        logits = (q[b].float() @ k.T) * sm_scale          # [H, s]
        m = logits.max(dim=-1, keepdim=True).values
        p = (logits - m).exp()
        outs.append((p @ v) / p.sum(-1, keepdim=True))
        lses.append((m.squeeze(-1) + p.sum(-1).log()))
    return torch.stack(outs), torch.stack(lses)


@pytest.mark.parametrize("num_kv_splits", [1, 2, 4])
def test_kernel_matches_naive_fp32(num_kv_splits):
    max_pages = (max(SEQ_LENS) + PAGE - 1) // PAGE
    n_slots = BATCH * max_pages * PAGE
    src = torch.randn(n_slots, NOPE + ROPE, device=DEVICE) * 0.3
    rows = quantize_row_ds_mla(src)

    # non-trivial page table: pages shuffled per batch
    perm = torch.randperm(BATCH * max_pages, device=DEVICE)
    req_to_token = perm.reshape(BATCH, max_pages).int()

    q = torch.randn(BATCH, HEADS, NOPE + ROPE, device=DEVICE, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(NOPE + ROPE)
    seq = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)

    o = torch.empty(BATCH, HEADS, NOPE, device=DEVICE, dtype=torch.float32)
    lse = torch.empty(BATCH, HEADS, device=DEVICE, dtype=torch.float32)
    attn_logits = torch.empty(BATCH, HEADS, num_kv_splits, NOPE + 1,
                              device=DEVICE, dtype=torch.float32)

    decode_attention_fwd_ds_mla(q, rows, o, lse, req_to_token, seq, attn_logits,
                                num_kv_splits, sm_scale, PAGE)

    ref_o, ref_lse = _naive_ref(q, rows, req_to_token, SEQ_LENS, sm_scale)
    cos = torch.nn.functional.cosine_similarity(
        o.flatten().float(), ref_o.flatten(), dim=0).item()
    assert cos > 0.99999, f"cosine {cos}"
    torch.testing.assert_close(o.float(), ref_o, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(lse.float(), ref_lse, rtol=2e-4, atol=2e-4)


def test_split_invariance():
    """Split-KV merge must not change the result (the GLM scratch-kernel
    lesson: an invalid merge poisons quality silently)."""
    max_pages = (max(SEQ_LENS) + PAGE - 1) // PAGE
    n_slots = BATCH * max_pages * PAGE
    rows = quantize_row_ds_mla(torch.randn(n_slots, NOPE + ROPE, device=DEVICE))
    req_to_token = (torch.arange(BATCH * max_pages, device=DEVICE)
                    .reshape(BATCH, max_pages).int())
    q = torch.randn(BATCH, HEADS, NOPE + ROPE, device=DEVICE)
    seq = torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32)
    sm_scale = 0.05
    outs = []
    for s in (1, 4):
        o = torch.empty(BATCH, HEADS, NOPE, device=DEVICE)
        lse = torch.empty(BATCH, HEADS, device=DEVICE)
        logits = torch.empty(BATCH, HEADS, s, NOPE + 1, device=DEVICE)
        decode_attention_fwd_ds_mla(q, rows, o, lse, req_to_token, seq, logits,
                                    s, sm_scale, PAGE)
        outs.append(o.clone())
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-4)
