"""
Unit tests for CalibrationEngine and PostTrainingQuantizer.
"""
import torch
import pytest
from compression_suite.quantization.calibration import CalibrationEngine
from compression_suite.quantization.quantizer import (
    quantize_tensor,
    dequantize_tensor,
    compute_sqnr,
    PostTrainingQuantizer,
    SimulatedPTQModel,
)
from compression_suite.models.registry import build_model


def test_quantize_dequantize_roundtrip():
    tensor = torch.randn(100, 100) * 5.0
    calibrator = CalibrationEngine(method="minmax", symmetric=True, q_min=-128, q_max=127)
    scale, zero_point = calibrator.calculate_scale_and_zero_point(tensor)

    q = quantize_tensor(tensor, scale, zero_point)
    assert q.dtype == torch.int8
    assert q.min() >= -128
    assert q.max() <= 127

    dequant = dequantize_tensor(q, scale, zero_point)
    assert dequant.shape == tensor.shape

    # SQNR for 8-bit quantization is typically between 30 dB and 50 dB
    sqnr = compute_sqnr(tensor, dequant)
    assert sqnr > 25.0


def test_percentile_calibration_clips_extreme_outliers():
    # Normal distribution with extreme outliers
    tensor = torch.randn(1000)
    tensor[0] = 500.0  # extreme positive outlier
    tensor[1] = -500.0 # extreme negative outlier

    cal_minmax = CalibrationEngine(method="minmax", symmetric=True)
    scale_minmax, _ = cal_minmax.calculate_scale_and_zero_point(tensor)

    cal_pct = CalibrationEngine(method="percentile", percentile=99.0, symmetric=True)
    scale_pct, _ = cal_pct.calculate_scale_and_zero_point(tensor)

    # Percentile calibration should ignore the +/- 500 outlier and produce a much tighter scale
    assert scale_pct < (scale_minmax * 0.1)


def test_simulated_ptq_model_forward():
    model = build_model("fast_convnet_student", num_classes=10)
    quantizer = PostTrainingQuantizer(method="percentile", dtype="int8")

    q_model = quantizer.quantize(model, use_native_torch=False)
    assert isinstance(q_model, SimulatedPTQModel)

    x = torch.randn(2, 3, 32, 32)
    out = q_model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()
