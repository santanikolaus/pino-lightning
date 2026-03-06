### NO – Darcy, Kolmogorov

Big whirls have little whirls,

That feed on their velocity,

And little whirls have lesser whirls,

And so on to viscosity.

— Lewis Fry Richardson

**documentation:**
data-only run 16x16 grid
14-36-47_r1

data 16x16 with 64x64 physics loss, 1:1 loss 
15-13-56_r2

data 16x16 with 64x64 physics loss, 1:0.1 loss 
16-22-49_r3

data 16x16 with 64x64 physics loss, 1:0.1 loss; mollifier 
20-18-44_r4

---

suggestion claude: mollifier, implemented

suggestion openai:
> Diagnosis
  - src/pde/darcy.py:12-57 hard-codes the PDE forcing to the unit-valued constant (PINO
  paper’s setup). The Darcy dataset you are training on does not use that forcing. I
  verified by applying the same operator to the raw ground-truth fields (~/data/darcy/
  darcy_train_16.pt) and observed residual magnitudes in the tens (center-averaged |
  R(u_true)| ≈ 4–6, extrema >100). Because the physics loss minimizes ||-div(a∇û_pred) −
  1||, it drives the network toward a solution that matches an incorrect PDE instead of
  the data distribution.
  - Training metrics confirm the conflict: run r1 (data only) reaches val_16_l2 ≈ 2
  (wandb/run-20260305_143657-70d3x9i2/files/wandb-summary.json), while r2/r3 with physics
  loss plateau near val_16_l2 ≈ 8/5.4 even though train_pde_loss becomes small (e.g.,
  run r2 reaches train_pde_loss_epoch ≈ 15.9 with tiny residual heatmaps). The model
  is “winning” on the wrong PDE and ignoring high-frequency structure seen in data-only
  training (tests/visual/figures/i2ul40fw_r2/power_spectrum_pred_vs_true.png).
  - Visual diagnostics highlight the same issue: for r2/r3 the PDE residual plots
  (tests/visual/figures/i2ul40fw_r2/pde_residual_batch_avg.png, tests/visual/figures/
  fdezehwe_r3/pde_residual_spatial.png) show predictions with near-zero residual, while
  the true fields have large structured residuals. This mismatch forces the network
  to attenuate the true modes (field comparisons in tests/visual/figures/i2ul40fw_r2/
  field_comparison.png) leading to the under-expressive spectra you noticed.

  Most Probable Cause

  The physics loss is enforcing the wrong PDE: the dataset’s forcing term varies
  spatially, but DarcyLoss assumes a constant unit source. Consequently, the physics term
  punishes the true solution and rewards smoother, low-frequency fields that satisfy f≡1,
  explaining why data-only training performs best.

  Actionable Next Step

  Compute or provide the correct forcing for each sample and use it inside the
  physics loss. One practical change: during batch preparation, denormalize the
  ground-truth pressure (similar to _denormalize_for_physics), evaluate f_true =
  DarcyLoss.pde._operator(u_true_phys, a) once per batch (without gradients), and
  minimize ||-div(a∇û_pred) − f_true||. Implement this in src/models/darcy_module.py:62-
  104 by passing f_true into an extended DarcyLoss (or subtracting it from raw_pde). Then
  re-run r2 with the same weights to isolate the impact.

  Next Steps

  - After wiring the correct forcing, re-run the PINO experiment (start with the lower
  PDE weight from r3) and compare validation power spectra/field plots; the physics term
  should now align with the data.
  - If the fix helps, sweep the PDE weight to find the best balance before considering
  further architectural changes.