import pytest
import torch

from src.datasets.transforms.normalizers import UnitGaussianNormalizer


def _masked_stats_like_impl(
    x: torch.Tensor, *, dim: list[int], mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Matches UnitGaussianNormalizer masked math (no eps inside std).
    m = mask.unsqueeze(0).to(dtype=x.dtype)  # (1, C, H, W)
    wx = x * m

    n_t = torch.sum(m, dim=dim, keepdim=True)
    if 0 in dim:
        n_t = n_t * x.shape[0]

    n_safe = n_t.clamp_min(1.0)

    mean = torch.sum(wx, dim=dim, keepdim=True) / n_safe
    sq_mean = torch.sum(wx**2, dim=dim, keepdim=True) / n_safe

    var = (sq_mean - mean**2).clamp_min(0.0)
    corr = n_t / (n_t - 1.0).clamp_min(1.0)
    std = torch.sqrt(var * corr)
    return mean, std, n_t


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((4, 2, 6, 5), [0, 2, 3]),
        ((3, 1, 4, 4), [0, 1, 2, 3]),
    ],
)
def test_update_mean_std_unmasked_matches_torch(shape, dim):
    x = torch.randn(*shape)
    n = UnitGaussianNormalizer(dim=dim, eps=1e-7)

    n.update_mean_std(x)

    assert torch.allclose(n.mean, x.mean(dim=dim, keepdim=True))
    assert torch.allclose(n.std, x.std(dim=dim, keepdim=True))


def test_partial_fit_unmasked_matches_batch_fit():
    x = torch.randn(6, 2, 4, 4)
    dim = [0, 2, 3]

    ref = UnitGaussianNormalizer(dim=dim, eps=1e-7)
    ref.fit(x)

    inc = UnitGaussianNormalizer(dim=dim, eps=1e-7)
    inc.partial_fit(x, batch_size=2)

    assert torch.allclose(inc.mean, ref.mean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(inc.std, ref.std, atol=1e-6, rtol=1e-5)
    assert inc.n_elements == ref.n_elements
    assert inc.n_elements == int(inc._n_elements_t.sum().item())


def test_update_mean_std_masked_semantics_and_shape_nonrectangular():
    B, C, H, W = 3, 2, 5, 4
    x = torch.randn(B, C, H, W)

    base = (torch.arange(H).unsqueeze(1) + torch.arange(W).unsqueeze(0)) % 2
    mask = base.unsqueeze(0).expand(C, H, W).to(torch.int64)  # (C, H, W)

    dim = [0, 2, 3]
    n = UnitGaussianNormalizer(dim=dim, mask=mask, eps=1e-7)
    n.update_mean_std(x.clone())

    exp_mean, exp_std, exp_n_t = _masked_stats_like_impl(x, dim=dim, mask=mask)

    assert n.mean.shape == exp_mean.shape
    assert n.std.shape == exp_std.shape

    assert torch.allclose(n.mean, exp_mean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(n.std, exp_std, atol=1e-6, rtol=1e-5)
    assert torch.allclose(n._n_elements_t, exp_n_t)
    assert n.n_elements == int(n._n_elements_t.sum().item())


def test_update_mean_std_masked_batch_correction_branch_sets_n_t():
    B, C, H, W = 4, 1, 6, 6
    x = torch.randn(B, C, H, W)

    mask = torch.zeros(C, H, W, dtype=torch.int64)
    mask[:, 1::2, ::3] = 1  # nontrivial count

    dim = [0, 2, 3]
    n = UnitGaussianNormalizer(dim=dim, mask=mask, eps=1e-7)
    n.update_mean_std(x)

    expected_n_t = mask.unsqueeze(0).to(x.dtype).sum(dim=dim, keepdim=True) * B
    assert torch.allclose(n._n_elements_t, expected_n_t)
    assert n.n_elements == int(expected_n_t.sum().item())


def test_partial_fit_masked_hits_incremental_path_and_matches_batch_fit_and_ref():
    B, C, H, W = 6, 1, 4, 5
    x = torch.randn(B, C, H, W)

    base = torch.zeros(H, W, dtype=torch.int64)
    base[::2, ::2] = 1
    mask = base.unsqueeze(0).expand(C, H, W)

    dim = [0, 2, 3]

    ref = UnitGaussianNormalizer(dim=dim, mask=mask, eps=1e-7)
    ref.update_mean_std(x.clone())

    exp_mean, exp_std, _ = _masked_stats_like_impl(x, dim=dim, mask=mask)
    assert torch.allclose(ref.mean, exp_mean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(ref.std, exp_std, atol=1e-6, rtol=1e-5)

    inc = UnitGaussianNormalizer(dim=dim, mask=mask, eps=1e-7)
    inc.partial_fit(x.clone(), batch_size=2)

    assert torch.allclose(inc.mean, ref.mean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(inc.std, ref.std, atol=1e-6, rtol=1e-5)
    assert inc.n_elements == int(inc._n_elements_t.sum().item())


def test_update_mean_std_dim_without_batch():
    x = torch.randn(4, 2, 6, 6)
    n = UnitGaussianNormalizer(dim=[2, 3], eps=1e-7)
    n.fit(x)

    assert n.mean.shape == (4, 2, 1, 1)
    assert n.std.shape == (4, 2, 1, 1)
    assert torch.allclose(n.mean, x.mean(dim=[2, 3], keepdim=True))
    assert torch.allclose(n.std, x.std(dim=[2, 3], keepdim=True))
    assert n.n_elements == int(n._n_elements_t.sum().item())


def test_masked_dim_without_batch():
    x = torch.randn(4, 1, 6, 6)

    mask = torch.ones(1, 6, 6, dtype=torch.int64)
    mask[:, ::2, :] = 0

    n = UnitGaussianNormalizer(dim=[2, 3], mask=mask, eps=1e-7)
    n.fit(x)

    assert n.mean.shape == (4, 1, 1, 1)
    assert n.std.shape == (4, 1, 1, 1)
    assert n.n_elements == int(n._n_elements_t.sum().item())


def test_partial_fit_zero_dim_guard_does_not_crash_or_change_state():
    x = torch.empty(0, 2, 4, 4)
    n = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)

    n.partial_fit(x, batch_size=2)

    assert n.n_elements == 0
    assert n.mean is None
    assert n.std is None
    assert int(n._n_elements_t.sum().item()) == 0


def test_transform_inverse_roundtrip():
    x = torch.randn(5, 2, 6, 6)
    dim = [0, 2, 3]
    n = UnitGaussianNormalizer(dim=dim, eps=1e-7)
    n.fit(x)

    y = n.transform(x)
    x2 = n.inverse_transform(y)
    assert torch.allclose(x2, x, atol=1e-5, rtol=1e-5)

    assert torch.allclose(n.forward(x), y)


def test_transform_inverse_roundtrip_masked():
    x = torch.randn(4, 1, 6, 6)

    mask = torch.ones(1, 6, 6, dtype=torch.int64)
    mask[:, ::2, ::2] = 0

    n = UnitGaussianNormalizer(dim=[0, 2, 3], mask=mask, eps=1e-7)
    n.fit(x)

    y = n.transform(x)
    x2 = n.inverse_transform(y)
    assert torch.allclose(x2, x, atol=1e-5, rtol=1e-5)


def test_from_dataset_returns_per_key_normalizers_and_values_match_fit():
    class _DS:
        def __init__(self, xs, ys):
            self._xs = xs
            self._ys = ys

        def __iter__(self):
            for x, y in zip(self._xs, self._ys):
                yield {"x": x, "y": y}

    xs = [torch.randn(1, 4, 4) for _ in range(3)]
    ys = [torch.randn(1, 4, 4) for _ in range(3)]
    ds = _DS(xs, ys)

    norms = UnitGaussianNormalizer.from_dataset(ds, dim=[0, 2, 3], keys=["x"])
    assert set(norms.keys()) == {"x"}
    n = norms["x"]

    # from_dataset iterates twice (collect keys, then fit). This test uses a
    # re-iterable dataset on purpose.
    assert n.mean is not None
    assert n.std is not None

    ref = UnitGaussianNormalizer(dim=[0, 2, 3], eps=n.eps)
    for s in xs:
        ref.partial_fit(s.unsqueeze(0), batch_size=1)

    assert torch.allclose(n.mean, ref.mean, atol=1e-6, rtol=1e-5)
    assert torch.allclose(n.std, ref.std, atol=1e-6, rtol=1e-5)
    assert n.n_elements == int(n._n_elements_t.sum().item())


def test_all_zero_mask_no_crash_and_transform_is_finite_and_semantics():
    x = torch.randn(3, 1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.int64)

    n = UnitGaussianNormalizer(dim=[0, 2, 3], mask=mask, eps=1e-7)
    n.fit(x)

    assert torch.allclose(n.mean, torch.zeros_like(n.mean))
    assert torch.allclose(n.std, torch.zeros_like(n.std))
    assert n.n_elements == 0
    assert int(n._n_elements_t.sum().item()) == 0

    y = n.transform(x)
    assert torch.isfinite(y).all()


def test_single_sample_effective_n_elements_std_is_zero_masked():
    x = torch.randn(1, 1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.int64)
    mask[:, 2, 3] = 1  # single kept position -> variance of one value is 0

    n = UnitGaussianNormalizer(dim=[0, 2, 3], mask=mask, eps=1e-7)
    n.fit(x)

    assert n.n_elements == 1
    assert int(n._n_elements_t.sum().item()) == 1
    assert torch.allclose(n.std, torch.zeros_like(n.std))


def test_device_roundtrip_cpu_cuda_if_available():
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
