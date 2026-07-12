import torch

from src.models.pdearena import Unet


def test_unet_forward_shape_and_finite_scalar_only():
    """Vorticity config (scalar=1, vector=0): (B,T_hist,C,H,W) -> (B,T_fut,C,H,W).

    Exercises the full down/middle/up path and the derived-insize reshape,
    the part of the port most at risk from deleting the 5 Fourier classes.
    """
    torch.manual_seed(0)
    model = Unet(
        n_input_scalar_components=1,
        n_input_vector_components=0,
        n_output_scalar_components=1,
        n_output_vector_components=0,
        time_history=4,
        time_future=1,
        hidden_channels=8,
        activation="gelu",
        norm=True,
    ).eval()
    x = torch.randn(2, 4, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 1, 32, 32)
    assert torch.isfinite(out).all()
