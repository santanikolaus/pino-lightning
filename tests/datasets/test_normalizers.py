import pytest
import torch

from src.datasets.transforms.normalizers import UnitGaussianNormalizer


def test_fit_matches_torch_mean_std():
    x = torch.randn(4, 2, 6, 5)
    dim = [0, 2, 3]
    n = UnitGaussianNormalizer(dim=dim, eps=1e-7)
    n.fit(x)

    assert torch.allclose(n.mean, x.mean(dim=dim, keepdim=True))
    assert torch.allclose(n.std, x.std(dim=dim, keepdim=True))


def test_fit_dim_without_batch():
    x = torch.randn(4, 2, 6, 6)
    n = UnitGaussianNormalizer(dim=[2, 3], eps=1e-7)
    n.fit(x)

    assert n.mean.shape == (4, 2, 1, 1)
    assert n.std.shape == (4, 2, 1, 1)
    assert torch.allclose(n.mean, x.mean(dim=[2, 3], keepdim=True))
    assert torch.allclose(n.std, x.std(dim=[2, 3], keepdim=True))


def test_transform_inverse_roundtrip():
    x = torch.randn(5, 2, 6, 6)
    dim = [0, 2, 3]
    n = UnitGaussianNormalizer(dim=dim, eps=1e-7)
    n.fit(x)

    y = n(x)
    x2 = n.inverse_transform(y)
    assert torch.allclose(x2, x, atol=1e-5, rtol=1e-5)


def test_transform_produces_approx_zero_mean_unit_std():
    x = torch.randn(100, 1, 8, 8)
    n = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
    n.fit(x)
    y = n(x)

    assert torch.allclose(y.mean(dim=[0, 2, 3]), torch.zeros(1), atol=0.01)
    assert torch.allclose(y.std(dim=[0, 2, 3]), torch.ones(1), atol=0.01)


def test_dim_as_int_is_wrapped_to_list():
    n = UnitGaussianNormalizer(dim=0)
    assert n.dim == [0]


def test_device_roundtrip():
    x = torch.randn(2, 1, 4, 4)
    n = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
    n.fit(x)

    if torch.cuda.is_available():
        n.cuda()
        assert n.mean.device.type == "cuda"
        assert n.std.device.type == "cuda"
        n.cpu()

    n.to(torch.device("cpu"))
    assert n.mean.device.type == "cpu"
    assert n.std.device.type == "cpu"
