### legacy run

python -m legacy.scripts.train_darcy \
  --opt.n_epochs 5 \
  --data.n_train 200 \
  --data.batch_size 4 \
  --data.test_batch_sizes "[4,4]" \
  --data.folder "$HOME/data/darcy"

yields:
Eval: 16_h1=0.4533, 16_l2=0.2898, 32_h1=0.5764, 32_l2=0.2923
[1] time=0.81, avg_loss=0.3961, train_err=1.5844
Eval: 16_h1=0.3860, 16_l2=0.2682, 32_h1=0.5452, 32_l2=0.2797
[2] time=0.78, avg_loss=0.3151, train_err=1.2603
Eval: 16_h1=0.3426, 16_l2=0.2413, 32_h1=0.5164, 32_l2=0.2546
[3] time=0.76, avg_loss=0.2755, train_err=1.1019
Eval: 16_h1=0.2999, 16_l2=0.2072, 32_h1=0.4932, 32_l2=0.2230
[4] time=0.75, avg_loss=0.2469, train_err=0.9877
Eval: 16_h1=0.2765, 16_l2=0.1880, 32_h1=0.4878, 32_l2=0.2058

PYTHONPATH=/Users/nick/Documents/JKU/pinorepository \
      python -i legacy/examples/data/plot_darcy_flow.py

  - Step 1 (Minimal Lightning skeleton) – under src/:
OK      - src/datasets/darcy_datamodule.py – wrap the existing DarcyDataset/load_darcy_flow_small, returning identical
  tensors.
OK      - src/models/darcy_module.py – simple LightningModule using legacy.neuralop.get_model, computing LpLoss,
  logging it, and configuring AdamW.
      - src/train.py – Hydra entry that instantiates the new DataModule/Module and runs a 1‑epoch/2‑batch sanity
OK  check. Only move on once this runs. check
  - Step 2 (Data surface parity) – still in src/datasets/: port normalization, patching, and resolution handling out
  of legacy glue so DarcyDataModule owns all preprocessing. Keep the model import legacy if that helps; the win is
  having data behavior centralized.
  - Step 3 (Trainer logic) – grow DarcyLitModule to absorb everything previously in
  legacy.neuralop.training.Trainer: H1/L2 combos, eval cadence, AMP (Lightning Trainer handles), callbacks for
  checkpoints/logging, and distributed flags via Lightning strategies. After this, the legacy Trainer isn’t
  referenced anywhere.
  - Step 4 (Model code) – either continue importing legacy.neuralop.models.fno.FNO or transplant it into src/models/
  fno.py; do the latter only after training parity is proven so you can compare outputs against the Step‑0 snapshot.
  - Step 5 (Detach legacy) – once shapes, losses, and eval metrics behave like the snapshot, start deleting legacy
  dependencies/configs and rely entirely on src/ + configs/.
