"""Re-apply the sm_86/W-low-bit patches that patch(1) cannot land cleanly.
Idempotent: every block checks before writing."""
import sys

SP = sys.argv[1] if len(sys.argv) > 1 else "/venv/main/lib/python3.12/site-packages"


def patch(path, old, new, tag):
    src = open(path).read()
    if new in src:
        print(f"  {tag}: already applied")
        return
    assert old in src, f"{tag}: anchor missing in {path}"
    open(path, "w").write(src.replace(old, new, 1))
    print(f"  {tag}: applied")


# 1+2. SITU whitelist in TritonExperts (2 sites -> replace_all style)
p = f"{SP}/vllm/model_executor/layers/fused_moe/experts/triton_moe.py"
src = open(p).read()
old = """    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,"""
new = """    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SITU,
            MoEActivation.SILU,"""
n = src.count(old)
if n:
    open(p, "w").write(src.replace(old, new))
print(f"  triton SITU: {n} sites patched")

# 3. SITU in HummingExpertsBase
patch(
    f"{SP}/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py",
    """        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,""",
    """        return activation in [
            MoEActivation.SITU,
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,""",
    "humming SITU",
)

# 4. wna16_marlin: is_transposed treats HUMMING as checkpoint-layout
patch(
    f"{SP}/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py",
    "        self.is_transposed = self.wna16_backend != WNA16MoEBackend.FLASHINFER_TRTLLM",
    """        self.is_transposed = self.wna16_backend not in (
            WNA16MoEBackend.FLASHINFER_TRTLLM,
            WNA16MoEBackend.HUMMING,
        )""",
    "wna16 is_transposed",
)

# 5. wna16_marlin: Humming wiring in process_weights_after_loading
patch(
    f"{SP}/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py",
    """        if converted is None:
            # In-place backends (e.g. Humming) are not wired through this
            # marlin-only method; fail clearly rather than unpacking None.
            raise NotImplementedError(
                f"{type(self).__name__} does not support the "
                f"{self.wna16_backend.value} MoE backend."
            )""",
    """        if converted is None:
            if self.wna16_backend != WNA16MoEBackend.HUMMING:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support the "
                    f"{self.wna16_backend.value} MoE backend."
                )
            assert self.experts_cls is not None
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            self.moe_kernel = make_wna16_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.wna16_backend,
                layer=layer,
                routing_tables=layer._expert_routing_tables(),
            )
            return""",
    "wna16 humming wiring",
)

# 6. int_wna16: QuantizationArgs branch in _humming_wna16_weight_schema
patch(
    f"{SP}/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
    """            "desc_act": quant_config.desc_act,
            "sym": quant_config.is_sym,
        }
    raise TypeError(""",
    """            "desc_act": quant_config.desc_act,
            "sym": quant_config.is_sym,
        }
    if isinstance(quant_config, QuantizationArgs):
        quant_type = getattr(quant_config.type, "value", quant_config.type)
        quant_strategy = getattr(quant_config.strategy, "value", quant_config.strategy)
        return {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "type": str(quant_type),
            "num_bits": quant_config.num_bits,
            "strategy": str(quant_strategy),
            "group_size": quant_config.group_size,
            "symmetric": quant_config.symmetric,
        }
    raise TypeError(""",
    "int_wna16 schema",
)
print("HAND_PATCHES_DONE")
