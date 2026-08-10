# SPDX-License-Identifier: Apache-2.0
"""Apply the fp8_ds_mla-on-sm86 integration patches to an installed vLLM tree.

Idempotent; every block verifies its anchor. Usage:
    python -m fp8kv_k3.apply_vllm_patches [site-packages-path]

Patch list (mirrors the research plan):
  P3a  TritonMLABackend.supported_kv_cache_dtypes += fp8_ds_mla
  P3b  TritonMLABackend.get_kv_cache_shape -> 656-wide uint8 rows for ds_mla
  P3c  SM89 gate carve-out: fp8_ds_mla is storage-only fp8 (software dequant)
  P5   forward_mqa dispatch -> fp8kv_k3.kernel.decode_attention_fwd_ds_mla
"""

import sys


def patch(path, old, new, tag):
    src = open(path).read()
    if new in src:
        print(f"  {tag}: already applied")
        return
    assert old in src, f"{tag}: anchor missing in {path}"
    open(path, "w").write(src.replace(old, new, 1))
    print(f"  {tag}: applied")


def main(sp: str) -> None:
    tm = f"{sp}/vllm/v1/attention/backends/mla/triton_mla.py"

    # ---- P3a: declare the dtype
    patch(
        tm,
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]''',
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]''',
        "P3a dtype list",
    )

    # ---- P3b: 656-wide page rows for ds_mla (mirror flashmla_sparse)
    patch(
        tm,
        "class TritonMLABackend(MLACommonBackend):",
        '''class TritonMLABackend(MLACommonBackend):
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # Packed 656-byte rows: 512 fp8 nope + 4 fp32 tile scales + 64 bf16
            # rope. Spec dtype is uint8, so elements == bytes.
            return (num_blocks, block_size, 656)
        return MLACommonBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )
''',
        "P3b cache shape",
    )

    # ---- P3c: carve fp8_ds_mla out of the SM89 native-fp8 gate
    src = open(tm).read()
    anchor = 'if self.kv_cache_dtype.startswith("fp8") and not ('
    fixed = ('if self.kv_cache_dtype.startswith("fp8") and '
             'self.kv_cache_dtype != "fp8_ds_mla" and not (')
    if fixed in src:
        print("  P3c sm89 gate: already applied")
    else:
        assert anchor in src, "P3c: gate anchor missing"
        open(tm, "w").write(src.replace(anchor, fixed, 1))
        print("  P3c sm89 gate: applied")

    # ---- P5: dispatch decode to the packed-row kernel
    patch(
        tm,
        """        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)""",
        """        if self.kv_cache_dtype == "fp8_ds_mla":
            # Packed 656B uint8 rows: bypass the element-typed path entirely
            # (its [..., :kv_lora_rank] slice would read raw BYTES as latents).
            from fp8kv_k3.kernel import decode_attention_fwd_ds_mla

            assert kv_c_and_k_pe_cache.dtype == torch.uint8
            assert kv_c_and_k_pe_cache.shape[-1] == 656
            PAGE_SIZE = kv_c_and_k_pe_cache.size(1)
            block_table = attn_metadata.decode.block_table
            seq_lens = attn_metadata.decode.seq_lens
            if not attn_metadata.causal:
                query_len = (
                    attn_metadata.num_decode_tokens // attn_metadata.num_decodes
                )
                if query_len > 1:
                    block_table = block_table.repeat_interleave(query_len, dim=0)
                    seq_lens = seq_lens.repeat_interleave(query_len)
            decode_attention_fwd_ds_mla(
                q,
                kv_c_and_k_pe_cache.reshape(-1, 656),
                o,
                lse,
                block_table,
                seq_lens,
                attn_logits,
                num_kv_splits,
                self.scale,
                PAGE_SIZE,
            )
            return o, lse

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)""",
        "P5 forward_mqa dispatch",
    )
    print("VLLM_PATCHES_DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/venv/main/lib/python3.12/site-packages")
