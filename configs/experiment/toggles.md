# Experiment Toggles

Every knob available in the experiment YAML files, what it controls, and what changing it does.
Organised by config group. Values shown are the **default** unless marked otherwise.

---

## `data` group  (`configs/data/darcy_binary.yaml`)

### `train_resolution` · default `11`
The grid size used for training samples. Source data is 421×421; it is subsampled to this
resolution before being fed to the model.

- `11` — low-resolution regime (paper §4.1 target); model must generalize to higher res at test time
- `61` — train at mid-resolution; model already sees fine structure, generalization is trivial
- Higher values train closer to the test distributions but require more memory and compute

### `input_coord_channels` · default `false`
Prepends vertex-centred x/y coordinate grids to the permeability input channel.

- `false` — model input is `(N, 1, H, W)` (permeability only)
- `true` — model input is `(N, 3, H, W)` (permeability + x-coord + y-coord); requires `model.data_channels: 3`

Coordinates span [0, 1] inclusive via `torch.linspace`. Enables resolution generalization:
the model can infer its position in the domain at any test resolution.

### `encode_input` / `encode_output` · default `true`
Whether to apply channel-wise Gaussian normalization to inputs / outputs.

- `true` — normalizes; `DefaultDataProcessor` wraps data with fitted mean/std
- `false` — raw values passed through; required to match the paper (which uses no normalization on the binary dataset)

### `batch_size` · default `16`
Mini-batch size. Paper uses `20`. Larger batches smooth gradients but reduce steps/epoch.

### `n_train` · default `1000`
Number of training samples drawn from the source file.

---

## `model` group  (`configs/model/fno.yaml`)

### `n_modes` · default `[20, 20]`
Number of Fourier modes kept in each spatial dimension after `rfft2`.

- `[20, 20]` — retains modes 0–19 in each axis
- `[5, 5]` — retains only modes 0–4; at `train_resolution=11`, rfft2 produces 6 bins
  (indices 0–5), so modes 0–4 all receive gradients from the data loss. Modes above
  `floor(train_resolution / 2)` get **zero gradient** from the data path — they stay at
  random init and inject noise at super-resolution test time. Capping at 5 eliminates
  this noise (run 2c breakthrough).

### `data_channels` · default `1`
Number of input channels the FNO lifting layer expects.

- `1` — permeability only
- `3` — permeability + x/y coords; must match `input_coord_channels: true`

### `hidden_channels` · default `64`
Width (channel dimension) of every FNO spectral convolution layer.
Increasing this raises model capacity and memory/compute cost.

### `n_layers` · default `4`
Depth of the FNO: number of stacked spectral convolution blocks.

### `lifting_channel_ratio` · default `2`
Controls the width of the lifting MLP that maps `data_channels → hidden_channels`.

- `2` — lifting hidden dim = `2 × hidden_channels`
- `0` — single linear lift (no hidden layer); matches the paper architecture

### `domain_padding` · default `0.0`
Fraction of zero-padding added in each spatial dimension before spectral convolutions
to reduce aliasing from periodic boundary assumption.

- `0.0` — no padding; required to match the paper
- `0.125` — 12.5% padding (default neuralop value); adds compute but can help for non-periodic problems

### `positional_embedding` · default `"none"` (neuralop default)
Built-in coordinate injection inside neuralop's FNO.

- `null` — disabled; use `input_coord_channels: true` instead (explicit coord concat at dataset level, which is what the paper does)
- Other values depend on neuralop version

### `use_channel_mlp` · default `true`
Whether to add a pointwise MLP after each spectral block (channel-mixing).

- `true` — extra parameters and non-linearity between Fourier layers
- `false` — matches paper architecture; removes capacity but reduces overfitting risk

### `channel_mlp_expansion` · default `0.5`
Hidden width of the channel MLP as a fraction of `hidden_channels`. Only used when `use_channel_mlp: true`.

---

## `loss` group

### `loss/data_only.yaml` vs `loss/pino.yaml`
Selected via `- override /loss: data_only` or `- override /loss: pino` in the experiment defaults.

- `data_only` — supervised L2 only; `pde_weight: 0.0`; no DarcyLoss built
- `pino` — data loss + PDE residual loss at `train_resolution`; `pde_weight: 0.1` (overrideable)

### `training` · default `l2`
Loss function for the data (supervised) term.

- `l2` — relative L2 norm (`LpLoss(d=2, p=2)`)
- `h1` — H1 Sobolev norm (penalises gradient errors too)

### `data_weight` · default `1.0`
Scale factor applied to the data loss before summing with PDE loss.
Paper PINO config uses `5.0` (`xy_loss` in their notation).

### `pde_weight` · default `0.1`
Scale factor applied to the PDE residual loss. `0.0` disables PDE loss entirely.
Paper PINO config uses `1.0` (`f_loss` in their notation).

### `bc_mollifier` · default `false` (data_only) / `true` (pino)
Whether to multiply predictions by `sin(πx)·sin(πy)·mollifier_scale` before computing
any loss. Enforces zero Dirichlet boundary conditions exactly.

- `false` — raw model output compared to labels; boundary values unconstrained
- `true` — prediction is `model_out × scale × sin(πx)sin(πy)`; boundaries are exactly zero
  at every resolution regardless of what the model outputs. Key ingredient of run 2c
  breakthrough (val_61 48% → 6.88%).

### `mollifier_scale` · default `0.001`
Amplitude of the BC mollifier. The model must learn to output ~1000× the true solution
magnitude in the interior to compensate; the mollifier then squashes boundaries to zero.

- `1.0` — full-scale; prediction and labels are at the same magnitude but BCs not enforced
- `0.001` — paper value; effective enforcement of zero BCs

### `forcing` · default `1.0`
Constant right-hand side of the Darcy PDE: `-div(a ∇u) = forcing`.
Must match the source data. For the 421×421 binary dataset the physical forcing is 1.0.

### `forcing_is_coeff_scaled` · default `false`
If `true`, the forcing term is `forcing × a(x)` (spatially varying).
Keep `false` for the standard constant-forcing Darcy problem.

---

## `trainer` group  (`configs/trainer/default.yaml`)

### `max_epochs` · default `500`
Total training epochs. Paper uses 500. Smoke tests use 2.

### `accelerator` · default `gpu`
PyTorch Lightning accelerator: `gpu` or `cpu`.

### `devices`
Which GPU(s) to use. Use `'[3]'` to pin to GPU index 3 (avoids conflicts with other users).

---

## `opt` group  (set via experiment override)

### `weight_decay` · default `1e-4`
L2 regularisation coefficient for AdamW.

- `0.0` — no regularisation; matches paper (tested in run 1d, no benefit found)

---

## Quick reference — paper-matching config

```yaml
data:
  encode_input: false
  encode_output: false
  input_coord_channels: true
  train_resolution: 11

model:
  data_channels: 3
  n_modes: [5, 5]           # critical: cap at train_res usable modes
  hidden_channels: 64
  n_layers: 4
  lifting_channel_ratio: 0
  domain_padding: 0.0
  positional_embedding: null
  use_channel_mlp: false    # paper arch (optional, run 2c omits this)

loss:
  bc_mollifier: true
  mollifier_scale: 0.001
  data_weight: 5.0          # PINO only
  pde_weight: 1.0           # PINO only
```
