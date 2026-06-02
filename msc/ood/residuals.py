from pathlib import Path

import yaml
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import NSVorticity

_cfg = yaml.safe_load(
    (Path(__file__).parent.parent / "configs/models.yaml").open())
_MODEL_CFG = _cfg["operator"]
_INF = _cfg["inference"]

N_TEST = _INF["n_test"]
OFFSET_TEST = _INF["offset_test"]
SUB_T = _INF["sub_t"]
TIME_SCALE = _INF["time_scale"]
TEMPORAL_PAD = _INF["temporal_pad"]


class ResidualDecomposer:
    """Loads a pretrained KF-FNO checkpoint and extracts per-term PDE residuals.

    Terms returned by extract():
        wt   — temporal:   ∂ω/∂t
        adv  — advection:  u·∇ω
        diff — diffusion:  −ν∇²ω
        Du   — aggregate:  wt + adv + diff  (== LHS of NS vorticity eq)

    Identity  wt + adv + diff == Du  is asserted on every trajectory.
    """

    def __init__(self,
                 ckpt_path: str,
                 train_re: int,
                 device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.ns = NSVorticity(re=train_re)
        self.model = self._load(ckpt_path)

    def _load(self, ckpt_path: str) -> torch.nn.Module:
        model = build_fno_kf(_MODEL_CFG)
        ckpt = torch.load(ckpt_path,
                          weights_only=False,
                          map_location=self.device)
        state = {
            k[len("model."):]: v
            for k, v in ckpt["state_dict"].items() if k.startswith("model.")
        }
        model.load_state_dict(state, strict=True)
        return model.to(self.device).eval()

    def extract(self, data_path: str) -> list[dict[str, Tensor]]:
        """Inference over N_TEST trajectories; returns one dict per trajectory.

        Each dict: {"Du": ..., "wt": ..., "adv": ..., "diff": ...}
        All tensors shape (1, S, S, T-2), on self.device.
        """
        loader = DataLoader(
            KFDataset(data_path,
                      n_samples=N_TEST,
                      offset=OFFSET_TEST,
                      sub_t=SUB_T),
            batch_size=1,
            shuffle=False,
        )
        results = []
        for batch in loader:
            ic = batch["x"].to(self.device)
            T = batch["y"].shape[-1]

            with torch.no_grad():
                pred = kf_forward(self.model,
                                  ic,
                                  T,
                                  time_scale=TIME_SCALE,
                                  temporal_pad=TEMPORAL_PAD)

            w = pred.squeeze(1)
            Du, (wt, adv, diff) = self.ns.residual(w)

            err = (wt + adv + diff - Du).abs().max().item()
            assert err < 1e-5, f"Identity failed: max_err={err:.2e}"

            results.append({"Du": Du, "wt": wt, "adv": adv, "diff": diff})

        return results

    @staticmethod
    def fft_power(w: Tensor) -> Tensor:
        """Spatial FFT power per frequency per time step.

        w: (B, S, S, T) → (B, S, S, T)
        """
        w_h = torch.fft.fft2(w, dim=[1, 2])
        return w_h.real**2 + w_h.imag**2
