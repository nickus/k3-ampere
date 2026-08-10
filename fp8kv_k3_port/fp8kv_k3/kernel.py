# SPDX-License-Identifier: Apache-2.0
"""Grouped split-KV decode over a packed fp8_ds_mla (656 B/token) cache.

Stage-1 mirrors vLLM's `_fwd_grouped_kernel_stage1` (triton_decode_attention.py)
control flow exactly — same grid, same paging walk, same online softmax, same
MLA `v = trans(k)` reuse — with the load path rewritten for the packed row:

  * NoPE: 512 fp8e4m3fn bytes -> in-register bit-math decode -> per-128 tile
    fp32 scale (read via a float32 view of the same buffer).
  * RoPE: 64 raw bf16 (read via a uint16 view; never scaled).

The cache is passed as THREE views of one allocation (uint8 / float32 /
uint16); rows are 656 B = 164 f32 = 328 u16, so all three views are exact.
Stage-2 (`_fwd_kernel_stage2`) is vendored verbatim from vLLM Apache-2.0.
"""

import torch
import triton
import triton.language as tl

from .dequant import dequant_bitmath_triton
from .layout import ROPE_U16_OFF, ROW_BYTES, SCALE_F32_OFF

ROW_F32 = ROW_BYTES // 4    # 164
ROW_U16 = ROW_BYTES // 2    # 328


@triton.jit
def _fwd_grouped_stage1_ds_mla(
    Q,
    KU8,          # uint8  view: rows of ROW_BYTES
    KF32,         # float32 view of the same buffer: rows of ROW_F32
    KU16,         # uint16 view of the same buffer: rows of ROW_U16
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,   # 512
    BLOCK_DPE: tl.constexpr,      # 64
    BLOCK_DV: tl.constexpr,       # 512
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Lv: tl.constexpr,             # 512
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)        # 0..511 nope element == byte
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_rope = tl.arange(0, BLOCK_DPE)        # 0..63
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0, cache_modifier=".ca")
    off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh
               + (BLOCK_DMODEL + offs_rope)[None, :])
    qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0, cache_modifier=".ca")

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_page_number = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch + offs_n // PAGE_SIZE,
                mask=mask_n, other=0, cache_modifier=".ca",
            ).to(tl.int64)  # page_number * row stride overflows int32 at rig scale
            # token slot index within the flat [num_blocks*page_size] row array
            tok = kv_page_number * PAGE_SIZE + (offs_n % PAGE_SIZE)

            # ---- NoPE: [BLOCK_DMODEL, BLOCK_N] fp8 bytes -> scaled q.dtype
            offs_k_bytes = tok[None, :] * ROW_BYTES + offs_d[:, None]
            kb = tl.load(KU8 + offs_k_bytes, mask=mask_n[None, :], other=0)
            k = dequant_bitmath_triton(kb).to(tl.float32)
            offs_scale = tok[None, :] * ROW_F32 + SCALE_F32_OFF + (offs_d[:, None] // 128)
            ks = tl.load(KF32 + offs_scale, mask=mask_n[None, :], other=1.0)
            k = (k * ks).to(q.dtype)

            qk = tl.dot(q, k)

            # ---- RoPE: [BLOCK_DPE, BLOCK_N] raw bf16 via uint16 view
            offs_rope_u16 = tok[None, :] * ROW_U16 + ROPE_U16_OFF + offs_rope[:, None]
            rb = tl.load(KU16 + offs_rope_u16, mask=mask_n[None, :], other=0)
            kpe = rb.to(tl.bfloat16, bitcast=True).to(q.dtype)
            qk += tl.dot(qpe, kpe)

            qk *= sm_scale
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            v = tl.trans(k)  # MLA: V is the dequantized NoPE block, reused

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh
                      + split_kv_id * stride_mid_os + offs_dv[None, :])
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=mask_h[:, None])
        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh
                        + split_kv_id * stride_mid_os + Lv)
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _fwd_kernel_stage2(  # vendored verbatim from vLLM triton_decode_attention.py
    Mid_O, o, lse, B_Seqlen,
    stride_mid_ob, stride_mid_oh, stride_mid_os,
    stride_obs, stride_oh, stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr, BLOCK_DV: tl.constexpr, Lv: tl.constexpr,
    OUTPUT_FP16: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv
    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os,
                         mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max
    result = acc / e_sum
    if OUTPUT_FP16:
        result = result.to(tl.float16)
    tl.store(o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
             result, mask=mask_d)
    tl.store(lse + cur_batch * stride_lse_bs + cur_head, e_max + tl.log(e_sum))


def decode_attention_fwd_ds_mla(
    q: torch.Tensor,            # [batch, q_heads, 576]
    kv_rows_u8: torch.Tensor,   # [num_slots, 656] uint8 (flat page rows)
    o: torch.Tensor,            # [batch, q_heads, 512]
    lse: torch.Tensor,          # [batch, q_heads]
    req_to_token: torch.Tensor,  # [batch, max_pages]
    b_seq_len: torch.Tensor,    # [batch]
    attn_logits: torch.Tensor,  # [batch, q_heads, splits, 512+1] fp32
    num_kv_splits: int,
    sm_scale: float,
    page_size: int,
    BLOCK_N: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
):
    assert kv_rows_u8.dtype == torch.uint8 and kv_rows_u8.shape[-1] == ROW_BYTES
    assert kv_rows_u8.is_contiguous()
    batch, q_heads = q.shape[0], q.shape[1]
    kf32 = kv_rows_u8.view(torch.int8).view(-1).view(torch.float32)
    ku16 = kv_rows_u8.view(torch.int8).view(-1).view(torch.uint16)

    BLOCK_H = 16 if q_heads > 16 else triton.next_power_of_2(max(q_heads, 1))
    grid = (batch, triton.cdiv(q_heads, min(BLOCK_H, q_heads)), num_kv_splits)
    _fwd_grouped_stage1_ds_mla[grid](
        q, kv_rows_u8.view(-1), kf32, ku16,
        sm_scale, req_to_token, b_seq_len, attn_logits,
        req_to_token.stride(0), q.stride(0), q.stride(1),
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        kv_group_num=q_heads, q_head_num=q_heads,
        BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=512,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size, Lv=512,
        num_warps=num_warps, num_stages=num_stages,
    )
    grid2 = (batch, q_heads)
    _fwd_kernel_stage2[grid2](
        attn_logits, o, lse, b_seq_len,
        attn_logits.stride(0), attn_logits.stride(1), attn_logits.stride(2),
        o.stride(0), o.stride(1), lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits, BLOCK_DV=512, Lv=512,
        num_warps=4, num_stages=2,
    )
