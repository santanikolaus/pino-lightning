import torch
from neuralop import LpLoss

from src.models.kf_module import KFLitModule, _get


class KFLitModuleChain(KFLitModule):
    """
    KFLitModule + window-chaining training signal.

    W1: standard forward + full PINO loss (data + PDE + IC + energy + angle).
    W2: second full-T pass from W1's predicted midpoint; data loss on overlap only.
    Gradient path: dL_chain/dpred2 * dpred2/dmid * dmid/dtheta  (no detach by default).
    Inference: inherited validation_step, one-shot, unchanged.

    chain config keys (under cfg.chain):
      m:          int | None  — handoff frame index (0-based in pred1 output); None -> T // 2
      weight:     float       — lambda on chain loss  (default 1.0)
      stop_grad:  bool        — detach mid before W2  (default False; True = stopgrad ablation)
    """

    def __init__(self, config):
        super().__init__(config)
        chain_cfg = _get(config, "chain")
        self.chain_m         = _get(chain_cfg, "m", None)
        self.chain_weight    = _get(chain_cfg, "weight", 1.0)
        self.chain_stop_grad = _get(chain_cfg, "stop_grad", False)
        self._lp = LpLoss(d=3, p=2, reduction="mean")

    def training_step(self, batch, batch_idx):
        ic     = batch["x"].to(self.device)
        target = batch["y"].to(self.device)
        if self.data_t_lo is not None and self.data_t_hi is not None:
            target = target[..., self.data_t_lo:self.data_t_hi]
        T = target.shape[-1]
        m = self.chain_m if self.chain_m is not None else T // 2

        # forward passes in AMP precision (fp16 safe for activations)
        pred1 = self(ic, T=T)          # (B, 1, S, S, T)
        mid   = pred1[:, 0, :, :, m]   # (B, S, S); 0-based in pred1 -> mirrors oracle_distill
        if self.chain_stop_grad:
            mid = mid.detach()
        pred2 = self(mid, T=T)         # (B, 1, S, S, T)

        # losses in fp32: PDE residual overflows fp16 (wavenumber^2 factors)
        with torch.autocast(device_type=self.device.type, enabled=False):
            losses1    = self.loss_fn(pred1.float(), target.float())
            chain_loss = self._lp.rel(
                pred2[:, 0, ..., :T - m].float(),   # (B, S, S, T-m)
                target[..., m:].float(),             # (B, S, S, T-m)
            )

        total = losses1["loss"] + self.chain_weight * chain_loss
        self.log("train_loss",       total,            prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_w1_loss",    losses1["loss"],  on_step=True, on_epoch=True)
        self.log("train_chain_loss", chain_loss,       on_step=True, on_epoch=True)
        self.log("train_data_loss",  losses1["data"],  on_step=True, on_epoch=True)
        self.log("train_pde_loss",   losses1["pde"],   on_step=True, on_epoch=True)
        self.log("train_ic_loss",    losses1["ic"],    on_step=True, on_epoch=True)
        return total

    # validation_step, configure_optimizers, forward — all inherited
    # NOTE: val_l2 is full L2, not late k<=7; run chain_gate.py post-training for wall metric
