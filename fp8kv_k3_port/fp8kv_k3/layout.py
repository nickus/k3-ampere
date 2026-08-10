# SPDX-License-Identifier: Apache-2.0
"""fp8_ds_mla row layout (656 B/token), pinned to vLLM's CUDA writer.

Byte map per token row (uint8 view, 656 bytes = 164 fp32 = 328 uint16):
  [  0, 512): 4 tiles x 128 fp8e4m3fn NoPE latents, tile t scaled by amax_t/448
  [512, 528): 4 fp32 tile scales           (fp32 elements [128, 132))
  [528, 656): 64 raw bf16 RoPE values      (uint16 elements [264, 328))
"""

import torch

from .dequant import dequant_bitmath_torch

NOPE = 512
ROPE = 64
TILE = 128
N_TILES = NOPE // TILE          # 4
SCALE_BYTE_OFF = 512
ROPE_BYTE_OFF = 528
ROW_BYTES = 656
SCALE_F32_OFF = SCALE_BYTE_OFF // 4    # 128
ROPE_U16_OFF = ROPE_BYTE_OFF // 2      # 264
FP8_MAX = 448.0


def quantize_row_ds_mla(src: torch.Tensor) -> torch.Tensor:
    """[N, 576] float (nope 512 | rope 64) -> [N, 656] uint8 rows."""
    assert src.shape[-1] == NOPE + ROPE
    n = src.shape[0]
    out = torch.zeros(n, ROW_BYTES, dtype=torch.uint8, device=src.device)
    nope = src[:, :NOPE].float().reshape(n, N_TILES, TILE)
    scales = nope.abs().amax(dim=-1) / FP8_MAX          # [N, 4]
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = (nope / scales[:, :, None]).to(torch.float8_e4m3fn)
    out[:, :NOPE] = q.reshape(n, NOPE).view(torch.uint8)
    out.view(torch.int8)[:, SCALE_BYTE_OFF:ROPE_BYTE_OFF] = (
        scales.to(torch.float32).view(torch.int8).reshape(n, 16)
    )
    out.view(torch.int8)[:, ROPE_BYTE_OFF:] = (
        src[:, NOPE:].to(torch.bfloat16).view(torch.int8).reshape(n, ROPE * 2)
    )
    return out


def dequantize_row_ds_mla(row: torch.Tensor,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[N, 656] uint8 -> [N, 576] `dtype`, the fp32 golden for kernels."""
    assert row.shape[-1] == ROW_BYTES and row.dtype == torch.uint8
    n = row.shape[0]
    scales = row.view(torch.int8)[:, SCALE_BYTE_OFF:ROPE_BYTE_OFF].reshape(
        n, N_TILES, 4).contiguous().view(torch.float32).reshape(n, N_TILES)
    nope = dequant_bitmath_torch(row[:, :NOPE], torch.float32).reshape(
        n, N_TILES, TILE) * scales[:, :, None]
    rope = row.view(torch.int8)[:, ROPE_BYTE_OFF:].reshape(
        n, ROPE, 2).contiguous().view(torch.bfloat16).reshape(n, ROPE)
    return torch.cat([nope.reshape(n, NOPE), rope.float()], dim=-1).to(dtype)
