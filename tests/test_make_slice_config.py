# SPDX-License-Identifier: Apache-2.0
"""Tests for the K3 slice-config generator.

The slice must preserve every layout-critical dimension of the real
Kimi K3 checkpoint (MLA head layout, KDA head_dim, quant format and its
ignore list) while shrinking everything that only scales capacity.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_slice_config import REAL_CONFIG_PATH, build_slice_config

REAL = json.loads(Path(REAL_CONFIG_PATH).read_text())["text_config"]


@pytest.fixture(scope="module")
def phase_a():
    return build_slice_config(phase="a")


@pytest.fixture(scope="module")
def phase_b():
    return build_slice_config(phase="b")


@pytest.fixture(scope="module")
def phase_c():
    return build_slice_config(phase="c")


def test_text_only_architecture(phase_a):
    # Text-only slice: same module path in vLLM, skips the vision tower.
    assert phase_a["architectures"] == ["KimiLinearForCausalLM"]
    assert phase_a["model_type"] == "kimi_linear"


def test_layer_pattern_one_mla_every_fourth(phase_a):
    la = phase_a["linear_attn_config"]
    assert phase_a["num_hidden_layers"] == 4
    assert la["kda_layers"] == [1, 2, 3]  # 1-indexed, mirrors real config
    assert la["full_attn_layers"] == [4]
    # Real config: layer % 4 == 0 -> full attention. Slice keeps the ratio.
    assert set(la["kda_layers"]) | set(la["full_attn_layers"]) == {1, 2, 3, 4}


def test_layout_critical_dims_match_real(phase_a):
    """These pin kernel layouts (fused cache-insert head 576, FA2 whitelist,
    upstream-tested KDA Triton shapes) and must equal production values."""
    for key in (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "q_lora_rank",
        "hidden_act",
        "moe_router_activation_func",
        "topk_method",
        "activation_situ_beta",
        "activation_situ_linear_beta",
        "mla_use_nope",
        "mla_use_output_gate",
        "latent_moe_use_norm",
        "first_k_dense_replace",
    ):
        assert phase_a[key] == REAL[key], key
    assert phase_a["linear_attn_config"]["head_dim"] == 128
    assert (
        phase_a["linear_attn_config"]["short_conv_kernel_size"]
        == REAL["linear_attn_config"]["short_conv_kernel_size"]
    )
    assert (
        phase_a["linear_attn_config"]["use_full_rank_gate"]
        == REAL["linear_attn_config"]["use_full_rank_gate"]
    )


def test_kda_heads_are_upstream_tested_shape(phase_a):
    # test_kda.py exercises H in {8,...}, D=128; fused-decode whitelist wants
    # {12,24,48,96} which we deliberately do NOT need (sm_86 uses Triton).
    assert phase_a["linear_attn_config"]["num_heads"] == 8
    assert phase_a["num_attention_heads"] == 8
    assert phase_a["num_key_value_heads"] == 8


def test_moe_shrunk_but_structured(phase_a):
    assert phase_a["num_experts"] == 8
    assert phase_a["num_experts_per_token"] == 2
    assert phase_a["num_shared_experts"] == REAL["num_shared_experts"]
    # multiples of 32 keep MXFP4 block packing valid in phase b
    assert phase_a["routed_expert_hidden_size"] % 32 == 0
    assert phase_a["moe_intermediate_size"] % 32 == 0


def test_vocab_and_special_tokens_are_real(phase_a):
    # Real tokenizer is reused verbatim -> ids must stay in range.
    assert phase_a["vocab_size"] == REAL["vocab_size"]
    assert phase_a["bos_token_id"] == REAL["bos_token_id"]
    assert phase_a["eos_token_id"] == REAL["eos_token_id"]


def test_phase_a_has_no_quant_and_no_mtp(phase_a):
    assert "quantization_config" not in phase_a
    assert phase_a["num_nextn_predict_layers"] == 0


def test_phase_b_quant_mirrors_real_checkpoint(phase_b):
    q = phase_b["quantization_config"]
    real_q = REAL["quantization_config"]
    assert q["quant_method"] == "compressed-tensors"
    assert q["format"] == "mxfp4-pack-quantized"
    # The ignore list is behavior: it decides what stays bf16 on the rig.
    assert q["ignore"] == real_q["ignore"]
    g = q["config_groups"]["group_0"]
    rg = real_q["config_groups"]["group_0"]
    assert g["weights"] == rg["weights"]
    assert g["targets"] == rg["targets"]


def test_phase_c_adds_one_mtp_layer(phase_c):
    assert phase_c["num_nextn_predict_layers"] == 1
    # everything else identical to phase a
    a = build_slice_config(phase="a")
    c = dict(phase_c)
    c["num_nextn_predict_layers"] = 0
    assert c == a


def test_configs_are_json_serializable_and_small(tmp_path, phase_a):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(phase_a, indent=2))
    assert json.loads(p.read_text()) == phase_a


def test_estimated_slice_size_fits_two_3090s(phase_a):
    """bf16 dummy-load footprint must be well under 2x24GB."""
    h = phase_a["hidden_size"]
    v = phase_a["vocab_size"]
    la = phase_a["linear_attn_config"]
    E = phase_a["num_experts"]
    reh = phase_a["routed_expert_hidden_size"]
    mi = phase_a["moe_intermediate_size"]
    params = 2 * v * h  # embed + head
    params += len(la["kda_layers"]) * (5 * h * la["num_heads"] * la["head_dim"])
    params += len(la["full_attn_layers"]) * (
        h * phase_a["q_lora_rank"]
        + h * (phase_a["kv_lora_rank"] + phase_a["qk_rope_head_dim"])
        + phase_a["q_lora_rank"]
        * phase_a["num_attention_heads"]
        * (phase_a["qk_nope_head_dim"] + phase_a["qk_rope_head_dim"])
        + phase_a["kv_lora_rank"]
        * phase_a["num_attention_heads"]
        * (phase_a["qk_nope_head_dim"] + phase_a["v_head_dim"])
    )
    moe_layers = phase_a["num_hidden_layers"] - phase_a["first_k_dense_replace"]
    params += moe_layers * (E + phase_a["num_shared_experts"]) * 3 * reh * mi
    params += phase_a["first_k_dense_replace"] * 3 * h * phase_a["intermediate_size"]
    bytes_bf16 = params * 2
    assert bytes_bf16 < 4e9, f"slice too fat: {bytes_bf16 / 1e9:.1f} GB"
