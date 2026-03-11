### NO – Darcy, Kolmogorov

Big whirls have little whirls,

That feed on their velocity,

And little whirls have lesser whirls,

And so on to viscosity.

— Lewis Fry Richardson

**documentation of test runs:**
data-only run 16x16 grid
14-36-47_r1

data 16x16 with 64x64 physics loss, 1:1 loss 
15-13-56_r2

data 16x16 with 64x64 physics loss, 1:0.1 loss 
16-22-49_r3

data 16x16 with 64x64 physics loss, 1:0.1 loss; mollifier 
20-18-44_r4

data 16x16 with 64x64 physics loss, 1:0.1 loss, forcing fix
17-01-06_r5

# todo. downsample from 421x421 for continuous forcing
---

suggestion claude: mollifier, implemented

> Diagnosis
Analysis: Root Cause of Underperformance

     The dataset stores normalized quantities, NOT the PINO paper's raw values

     The Darcy dataset stores:
     - a ∈ {0, 1} — a binary indicator, not the physical permeability {3, 12}
     - u — the solution scaled by K ≈ 32.32 from physical units

     I verified this empirically:

     ┌───────────────────┬───────────────────┬───────────────────────┬───────────────┐
     │      Region       │ -div(a·∇u) stored │   With α=3+9a, u/K    │ Paper expects │
     ├───────────────────┼───────────────────┼───────────────────────┼───────────────┤
     │ Deep a=1 interior │ 2.6936 ± 0.005    │ 1.0000 ± 0.04         │ 1.0           │
     ├───────────────────┼───────────────────┼───────────────────────┼───────────────┤
     │ Deep a=0 interior │ 0.0000            │ 4.5 (wrong K for a=0) │ 1.0           │
     └───────────────────┴───────────────────┴───────────────────────┴───────────────┘

     The stored data satisfies: -div(a·∇u) = C·a where C = 2.6936 (not f = 1).

     Your DarcyLoss uses f = 1, which means:
     - In a=1 regions: target is 1, actual is 2.69 → model is penalized for being
   correct
     - In a=0 regions: target is 1, actual is 0 → model is penalized for being correct

     This is why adding PDE loss hurts: the PDE loss drives the model toward the wrong
     solution.

     Visual evidence confirms this

     - r1 (data only): good field quality, errors ~0.1
     - r2 (PDE 1:1, f=1): fields washed out to near-zero — PDE dominates with wrong
   target
     - r3 (PDE 1:0.1, f=1): still harmed — wrong PDE still conflicts with data
     - r4 (PDE 1:0.1, mollifier): neutral — mollifier zeroes out enough of the loss that
   it
     can't harm much, but it can't help either

     The ratio problem

     The Laplacian ratio between a=0 and a=1 regions is 18:1, not 4:1 as expected for
   {3,
     12}. This means the dataset uses a different coefficient mapping than the paper's
   {3,
     12}. We can't simply transform a_stored → 3 + 9·a and get correct results
   everywhere.

     ---
     Recommended Next Step: Fix the forcing to f = C·a

     Change: In DarcyLoss, replace the constant forcing f=1 with f(x) = C · a(x), where
   C ≈
     2.6936.

     Why this is the right isolated step:
     1. It fixes the most critical issue — the PDE loss is currently actively harmful
     2. It's a ~5 line code change, easy to verify
     3. Testable: re-run with r4 settings (pde_weight=0.1, mollifier) and compare val_l2
   to
     r1
     4. If val_l2 improves over r1, the forcing was the primary problem

     Limitation: With a ∈ {0, 1}, the PDE constraint is zero in a=0 regions (no physics
     guidance there). This is inherent to the stored binary coefficient. The second
   forward
     pass (suggestion 1) is the second priority fix after this one.

     Expected outcome: PDE loss should now help rather than hurt — the model gets
   correct
     physics guidance in a=1 regions and no conflicting signal in a=0 regions.

     Shall I implement this fix?

   ✻ Worked for 16m 38s

   ⏺ Background command "Inspect actual permeability values across resolutions"
   completed
   (exit code 0)

   ⏺ Read 1 file (ctrl+o to expand)

   ⏺ This changes the picture significantly. The 421×421 data stores the raw continuous
     coefficient (range [0.45, 2.92]) and physical solution (max 0.086), while the 16/64
     data stores binarized a ∈ {0, 1} and scaled u (max 1.6). These are different
     representations.

     This confirms the data pipeline: the raw Darcy data (421×421) has continuous
     permeability and physical-scale u, but the lower-resolution files apply
   thresholding
     and normalization.

     This doesn't change my recommendation — fixing the forcing to f = C·a is still the
     right next step for the 16/64 data format currently in use. But it does suggest a
     longer-term opportunity: loading the raw 421×421 coefficient at 64×64 (subsampled)
   for
     the PDE loss, which would give a continuous coefficient and more accurate physics.

     Shall I implement the forcing fix?

...

┌───────────┬────────────────┬────────────────────┬──────────────────────┐
  │  Metric   │ r1 (data-only) │ r4 (wrong forcing) │ r5 (correct forcing) │
  ├───────────┼────────────────┼────────────────────┼──────────────────────┤
  │ val_16_l2 │ ~2.0           │ ~2.0               │ 1.99                 │
  ├───────────┼────────────────┼────────────────────┼──────────────────────┤
  │ val_64_l2 │ —              │ —                  │ 1.39                 │
  └───────────┴────────────────┴────────────────────┴──────────────────────┘

  The corrected forcing is essentially neutral — val_16_l2 ≈ 2.0, same as the data-only
  baseline. The PDE loss isn't hurting anymore (which the mollifier already achieved in
  r4), but it's also not helping.

  The train_loss_epoch ≈ 9.0 with data_weight=1.0 and pde_weight=0.1 suggests the raw PDE
   loss is still very large (~70), meaning the PDE residual remains high even with the
  correct forcing. This points to the bicubic upsampling issue as the next bottleneck:
  the model produces 16×16 output which gets bicubically upsampled to 64×64, and the
  finite-difference PDE residual on that smooth interpolated field may not provide useful
   gradients back to the FNO weights.

  The PINO paper's approach is to evaluate the FNO at the higher resolution natively
  (FNOs are discretization-convergent), not upsample. This is the second LLM suggestion
  and the logical next isolated step.

  Should I implement the native high-res forward pass for PDE loss evaluation? That would
   mean:
  1. During training, run the FNO forward pass at pde_resolution (64×64) on the input a
  (also upsampled to 64×64 or loaded at that resolution) to get a native high-res
  prediction
  2. Compute PDE loss on that native prediction
  3. Keep the data loss at 16×16 as before
..
Section 4.1 (p.12) — the core setup for Darcy:

  "the training data for the Darcy equation setting is at the spatial resolution of 11 ×
  11 and the J_pde is imposed in 61 × 61 resolution. We use 1000 low-resolution data and
  the same 1000 PDE instances."

  Section 1.1 (p.2) — the high-level approach:

  "ours is the first to incorporate PDE constraints at a higher resolution as a remedy
  for low resolution training data."

  "a unique feature that PINO enjoys over other hybrid learning methods [58, 57, 20] is
  its ability to incorporate data and PDE loss functions at different resolutions."

  Appendix A.2 (p.22) — Darcy implementation:

  "All the models are trained on resolution 61 × 61 and evaluated on resolution 61 × 61."

  This last quote is actually the crucial detail. It says the models are trained on        resolution 61×61. Combined with Table 1 which says data is at 11×11, this means:
  - The data pairs (a, u) are at 11×11
  - The FNO is trained/evaluated at 61×61 — the input a is provided at 61×61 for the PDE
  loss
  - The FNO produces a genuine 61×61 output which is then checked against the PDE

  The paper does not interpolate a low-res FNO output. It feeds the 61×61 input into the
  FNO and gets a 61×61 output natively, exploiting discretization convergence. For the
  data loss, it presumably downsamples or evaluates the FNO at 11×11 to compare against
  the 11×11 labels.

  Your analysis is correct. The current implementation's bicubic upsampling of the 16×16   output is fundamentally different from the paper's approach of running the FNO at
  61×61. The fix is a second forward pass at high resolution for the PDE loss.