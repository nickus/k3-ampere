# SPDX-License-Identifier: Apache-2.0
"""Patch 2: 656B row layout pinned by positive, negative and bit-exact tests."""

import torch

from fp8kv_k3.layout import (
    NOPE,
    ROPE,
    ROPE_BYTE_OFF,
    ROW_BYTES,
    SCALE_BYTE_OFF,
    dequantize_row_ds_mla,
    quantize_row_ds_mla,
)

from .conftest import DEVICE

torch.manual_seed(7)


def _rows(n=64):
    return torch.randn(n, NOPE + ROPE, device=DEVICE) * 0.5


def test_roundtrip_within_e4m3_tile_error():
    src = _rows()
    rows = quantize_row_ds_mla(src)
    assert rows.shape[-1] == ROW_BYTES
    back = dequantize_row_ds_mla(rows)
    cos = torch.nn.functional.cosine_similarity(back[:, :NOPE], src[:, :NOPE], dim=-1)
    assert cos.min().item() > 0.999, f"nope cosine {cos.min().item()}"
    rel = (back[:, :NOPE] - src[:, :NOPE]).abs().max() / src[:, :NOPE].abs().max()
    assert rel.item() < 2 ** -3


def test_rope_bytes_bit_exact():
    src = _rows()
    rows = quantize_row_ds_mla(src)
    back = dequantize_row_ds_mla(rows)
    want = src[:, NOPE:].to(torch.bfloat16)
    assert torch.equal(back[:, NOPE:].to(torch.bfloat16), want), \
        "RoPE region must be raw bf16 passthrough — any error means we quantized it"


def test_negative_scales_at_tail_breaks_decode():
    """Pin the scale offsets: a packer that puts scales at the TAIL must
    disagree with the golden decoder."""
    src = _rows(8)
    rows = quantize_row_ds_mla(src)
    corrupted = rows.clone()
    scale_bytes = rows[:, SCALE_BYTE_OFF:ROPE_BYTE_OFF].clone()
    corrupted[:, SCALE_BYTE_OFF:ROPE_BYTE_OFF] = rows[:, ROW_BYTES - 16:]
    corrupted[:, ROW_BYTES - 16:] = scale_bytes
    back = dequantize_row_ds_mla(corrupted)
    good = dequantize_row_ds_mla(rows)
    assert not torch.allclose(back[:, :NOPE], good[:, :NOPE]), \
        "moving the scales must be detected — offsets are load-bearing"


def test_zero_rows_survive():
    src = torch.zeros(4, NOPE + ROPE, device=DEVICE)
    back = dequantize_row_ds_mla(quantize_row_ds_mla(src))
    assert torch.equal(back, src)
