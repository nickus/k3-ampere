# SPDX-License-Identifier: Apache-2.0
"""Patch 1: bit-exactness of the transplanted fp8e4m3fn software decode."""

import pytest
import torch
import triton
import triton.language as tl

from fp8kv_k3.dequant import dequant_bitmath_torch, dequant_bitmath_triton

from .conftest import DEVICE


@triton.jit
def _sweep_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    b = tl.load(in_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, dequant_bitmath_triton(b), mask=m)


def _golden():
    return (
        torch.arange(256, dtype=torch.uint8, device=DEVICE)
        .view(torch.float8_e4m3fn)
        .to(torch.float16)
    )


def _assert_matches_golden(got, want):
    """Bit-exact on the 254 finite fp8e4m3fn values; is-NaN on the two NaN
    encodings (0x7F/0xFF). The exact NaN PAYLOAD is not contractual — torch's
    own cast differs across devices (0x7F80 on CPU, 0x7FFF on CUDA) — and NaN
    never appears in real KV data. Any NaN out is correct."""
    finite = ~(torch.isnan(want))
    g = got.view(torch.uint16).to(torch.int32)
    w = want.view(torch.uint16).to(torch.int32)
    assert torch.equal(g[finite], w[finite])
    assert torch.isnan(got[~finite]).all()


def test_torch_twin_bit_exact_all_256():
    got = dequant_bitmath_torch(torch.arange(256, dtype=torch.uint8, device=DEVICE))
    _assert_matches_golden(got, _golden())


def test_triton_kernel_bit_exact_all_256():
    u8 = torch.arange(256, dtype=torch.uint8, device=DEVICE)
    out = torch.empty(256, dtype=torch.float16, device=DEVICE)
    _sweep_kernel[(1,)](u8, out, 256, BLOCK=256)
    _assert_matches_golden(out, _golden())


@pytest.mark.parametrize("byte,mag", [(0x7F, 0x7F80), (0xFF, 0xFF80)])
def test_nan_encodings_sign_preserved(byte, mag):
    u8 = torch.tensor([byte], dtype=torch.uint8, device=DEVICE)
    assert dequant_bitmath_torch(u8).view(torch.uint16).item() == mag
