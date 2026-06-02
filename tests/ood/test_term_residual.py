import numpy as np
import pytest
import torch

from msc.ood.residuals import (
    ResidualDecomposer,
    N_TEST,
    OFFSET_TEST,
    SUB_T,
    TIME_SCALE,
    TEMPORAL_PAD,
)
from msc.ood.residual_analysis import ResidualAnalysis


@pytest.mark.parametrize(
    "B,S,T",
    [
        pytest.param(1, 8, 4, id="single_small"),
        pytest.param(3, 16, 8, id="batch_medium"),
        pytest.param(2, 32, 1, id="single_time_step"),
    ],
)
def test_fft_power_output_shape(B, S, T):
    w = torch.randn(B, S, S, T)
    out = ResidualDecomposer.fft_power(w)
    assert out.shape == (B, S, S, T)


@pytest.mark.parametrize(
    "B,S,T",
    [
        pytest.param(1, 4, 4, id="single_small"),
        pytest.param(3, 16, 6, id="batch_medium"),
        pytest.param(2, 8, 1, id="single_time_step"),
    ],
)
def test_fft_power_parseval(B, S, T):
    # Parseval: sum of |FFT|² over all frequencies == S² * spatial L2
    torch.manual_seed(42)
    w = torch.randn(B, S, S, T)
    got = ResidualDecomposer.fft_power(w).sum(dim=[1, 2])
    expected = (S * S) * (w ** 2).sum(dim=[1, 2])
    assert torch.allclose(got, expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize(
    "B,T",
    [
        pytest.param(1, 4, id="single_trajectory"),
        pytest.param(5, 10, id="batch"),
        pytest.param(3, 1, id="single_time_step"),
    ],
)
def test_time_mean_output_shape(B, T):
    power = torch.randn(B, T)
    out = ResidualAnalysis.time_mean(power.numpy())
    assert out.shape == (B,)


@pytest.mark.parametrize(
    "B,T",
    [
        pytest.param(1, 4, id="single_trajectory"),
        pytest.param(4, 8, id="batch"),
        pytest.param(2, 1, id="single_time_step"),
    ],
)
def test_time_mean_correctness(B, T):
    torch.manual_seed(7)
    power = torch.randn(B, T)
    got = ResidualAnalysis.time_mean(power.numpy())
    expected = power.numpy().mean(axis=-1)
    assert np.allclose(got, expected, rtol=1e-6, atol=0.0)


def test_module_constants_match_yaml():
    assert N_TEST == 40
    assert OFFSET_TEST == 260
    assert SUB_T == 2
    assert TIME_SCALE == 1.0
    assert TEMPORAL_PAD == 5
