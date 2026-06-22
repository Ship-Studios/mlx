import math

import pytest

from mlx_runner import memory


def test_format_bytes():
    assert memory.format_bytes(None) == "unknown"
    assert memory.format_bytes(0) == "0 B"
    assert memory.format_bytes(512) == "512 B"
    assert memory.format_bytes(1024) == "1.00 KiB"
    assert memory.format_bytes(1024 ** 3) == "1.00 GiB"
    # 7e9 bytes ~ 6.52 GiB
    assert memory.format_bytes(7_000_000_000).endswith("GiB")


def test_parse_param_count():
    assert memory.parse_param_count("7B") == 7_000_000_000
    assert memory.parse_param_count("1.5B") == 1_500_000_000
    assert memory.parse_param_count("350M") == 350_000_000
    assert memory.parse_param_count("7e9") == 7_000_000_000
    assert memory.parse_param_count("7000000000") == 7_000_000_000
    assert memory.parse_param_count("8b") == 8_000_000_000
    assert memory.parse_param_count(1234) == 1234


def test_parse_param_count_invalid():
    with pytest.raises(ValueError):
        memory.parse_param_count("")
    with pytest.raises(ValueError):
        memory.parse_param_count("abc")
    with pytest.raises(ValueError):
        memory.parse_param_count("-5B")


def test_bytes_per_weight_dtypes():
    assert memory.bytes_per_weight(dtype="float16") == 2.0
    assert memory.bytes_per_weight(dtype="bf16") == 2.0
    assert memory.bytes_per_weight(dtype="float32") == 4.0
    with pytest.raises(ValueError):
        memory.bytes_per_weight(dtype="nonsense")


def test_bytes_per_weight_quantized():
    # 4-bit, group 64: 0.5 + 4/64 = 0.5625 bytes/param
    assert memory.bytes_per_weight(quant_bits=4, group_size=64) == pytest.approx(0.5625)
    # 8-bit, group 64: 1.0 + 0.0625
    assert memory.bytes_per_weight(quant_bits=8, group_size=64) == pytest.approx(1.0625)
    with pytest.raises(ValueError):
        memory.bytes_per_weight(quant_bits=4, group_size=0)
    with pytest.raises(ValueError):
        memory.bytes_per_weight(quant_bits=0)


def test_estimate_weights_memory():
    # 1B params fp16 = 2 GB
    assert memory.estimate_weights_memory(1_000_000_000, dtype="float16") == 2_000_000_000
    # 7B 4-bit ~ 3.9375 GB
    w = memory.estimate_weights_memory(7_000_000_000, quant_bits=4, group_size=64)
    assert w == int(7_000_000_000 * 0.5625)
    with pytest.raises(ValueError):
        memory.estimate_weights_memory(0)


def test_estimate_kv_cache_memory():
    # 2 * 32 layers * 8 kv heads * 128 head_dim * 1024 seq * 2 bytes
    expected = 2 * 32 * 8 * 128 * 1024 * 2
    assert memory.estimate_kv_cache_memory(32, 8, 128, 1024) == expected
    # quantized kv (8-bit) halves it
    assert memory.estimate_kv_cache_memory(32, 8, 128, 1024, kv_bits=8) == expected // 2
    with pytest.raises(ValueError):
        memory.estimate_kv_cache_memory(0, 8, 128, 1024)


def test_estimate_model_memory_without_kv():
    est = memory.estimate_model_memory(1_000_000_000, dtype="float16", overhead_fraction=0.05)
    assert est.weights_bytes == 2_000_000_000
    assert est.kv_cache_bytes == 0
    assert est.overhead_bytes == int(2_000_000_000 * 0.05)
    assert est.total_bytes == est.weights_bytes + est.overhead_bytes
    d = est.to_dict()
    assert d["total_bytes"] == est.total_bytes


def test_estimate_model_memory_with_kv():
    est = memory.estimate_model_memory(
        7_000_000_000,
        quant_bits=4,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        seq_len=2048,
    )
    assert est.kv_cache_bytes > 0
    assert est.total_bytes > est.weights_bytes


def test_check_fit():
    est = memory.estimate_model_memory(1_000_000_000, dtype="float16")  # 2GB + 5%
    fit = memory.check_fit(est, available_bytes=8_000_000_000, safety_fraction=0.9)
    assert fit.fits is True
    assert fit.headroom_bytes > 0

    tight = memory.check_fit(est, available_bytes=2_000_000_000, safety_fraction=0.9)
    assert tight.fits is False
    assert tight.headroom_bytes < 0

    # accepts a raw byte count too
    raw = memory.check_fit(1_000, available_bytes=10_000)
    assert raw.fits is True

    with pytest.raises(ValueError):
        memory.check_fit(est, 1_000, safety_fraction=1.5)


def test_recommend_quantization_picks_highest_quality_that_fits():
    # 7B model, ~5GB budget -> fp16 (14GB) and 8-bit (~7.4GB) don't fit, 4-bit (~3.9GB) does.
    rec = memory.recommend_quantization(
        7_000_000_000, available_bytes=5_500_000_000, safety_fraction=1.0
    )
    assert rec["fits"] is True
    assert rec["quant_bits"] == 4


def test_recommend_quantization_full_precision_when_roomy():
    rec = memory.recommend_quantization(
        1_000_000_000, available_bytes=64_000_000_000, safety_fraction=0.9
    )
    assert rec["quant_bits"] is None
    assert rec["fits"] is True


def test_recommend_quantization_nothing_fits():
    rec = memory.recommend_quantization(
        70_000_000_000, available_bytes=1_000_000_000, safety_fraction=0.9
    )
    assert rec["fits"] is False
    assert rec["quant_bits"] == 2
