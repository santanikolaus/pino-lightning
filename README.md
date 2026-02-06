### legacy run

python -m legacy.scripts.train_darcy \
  --opt.n_epochs 5 \
  --data.n_train 200 \
  --data.batch_size 4 \
  --data.test_batch_sizes "[4,4]" \
  --data.folder "$HOME/data/darcy"

yields: 


PYTHONPATH=/Users/nick/Documents/JKU/pinorepository \
      python -i legacy/examples/data/plot_darcy_flow.py

