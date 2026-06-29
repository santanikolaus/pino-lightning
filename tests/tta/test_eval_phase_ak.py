"""Tests for eval_phase_ak: persistence floor and phase-vs-raw GT equivalence."""
import numpy as np
import torch

from msc.tta.eval import cheb_bins, energy_phase_band


def _rand_vorticity(B=1, S=16, T=8, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(B, S, S, T, generator=rng)


def test_persistence_ak_bounded():
    """IC phase frozen across T: A_k is a cosine-based quantity in [-1, 1]."""
    S, T = 16, 8
    device = torch.device("cpu")
    raw_gt = _rand_vorticity(S=S, T=T)
    ic = raw_gt[..., 0]                                           # (1, S, S)
    pred = ic.unsqueeze(-1).expand(-1, -1, -1, T).clone()         # (1, S, S, T)
    kinf = cheb_bins(S, device)
    n_bands = S // 2 + 1
    _, e_gt, e_cos = energy_phase_band(pred, raw_gt, kinf, n_bands)
    eps = 1e-30
    ak = e_cos.sum(1) / (e_gt.sum(1) + eps)
    assert np.all(ak >= -1.0 - 1e-6) and np.all(ak <= 1.0 + 1e-6)


def test_phase_normalisation_preserves_in_band_angles():
    """Phase normalisation sets |Fh|=1 per in-band mode but leaves arg(Fh) unchanged.

    For every in-band mode with nonzero amplitude, cos(angle_raw - angle_phase) == 1.
    This is the invariant that makes phase-file GT equivalent to raw GT for angle
    comparisons (only the energy weights differ).
    """
    from scripts.materialize_phase_k7 import phase_normalize

    S, T, kmax = 16, 8, 7
    raw_gt   = _rand_vorticity(S=S, T=T)                          # (1, S, S, T)
    phase_gt = phase_normalize(raw_gt, kmax)                      # (1, S, S, T)

    Fg_raw   = torch.fft.fft2(raw_gt,   dim=(1, 2))               # (1, S, S, T) complex
    Fg_phase = torch.fft.fft2(phase_gt, dim=(1, 2))

    # per-mode cosΔφ between raw and phase-normalised GT
    cos = (Fg_phase * Fg_raw.conj()).real / (Fg_phase.abs() * Fg_raw.abs() + 1e-12)

    device = torch.device("cpu")
    kinf = cheb_bins(S, device)
    eps_amp = 1e-6

    for k in range(1, kmax + 1):
        mask = (kinf == k)                                        # (S, S)
        amp  = Fg_raw[0][mask].abs()                              # (n_modes, T)
        cos_k = cos[0][mask]                                      # (n_modes, T)
        well_conditioned = amp > eps_amp
        if well_conditioned.any():
            mean_cos = cos_k[well_conditioned].mean().item()
            assert mean_cos > 0.999, (
                f"k={k}: mean cosΔφ={mean_cos:.6f} — phase normalisation changed angles"
            )


def test_varb_coarse_channel_changes_result():
    """Passing a non-zero coarse channel to kf_forward changes the output
    vs zero coarse, confirming the channel is wired through."""
    from src.models.kf_fno import kf_forward, build_fno_kf
    from msc.tta import setup as tta_setup

    device = torch.device("cpu")
    cfg = dict(tta_setup.MODEL_CFG)
    cfg["data_channels"] = 5
    model = build_fno_kf(cfg).to(device).eval()

    S, T = 16, 8
    torch.manual_seed(0)
    ic     = torch.randn(1, S, S)
    coarse = torch.randn(1, S, S, T)
    zeros  = torch.zeros(1, S, S, T)

    with torch.no_grad():
        out_real = kf_forward(model, ic, T, coarse_traj=coarse).squeeze(1)
        out_zero = kf_forward(model, ic, T, coarse_traj=zeros).squeeze(1)

    assert not torch.allclose(out_real, out_zero), \
        "coarse channel has no effect on output — wiring broken"
