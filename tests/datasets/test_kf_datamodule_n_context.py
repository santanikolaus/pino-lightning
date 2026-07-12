import numpy as np

from src.datasets.kf_datamodule import KFDataModule


def test_n_context_reaches_val_split(tmp_path):
    """n_context must reach the val dataset, not only train.

    Validation/eval warmup seeds the rollout from batch["ctx"]; a datamodule
    that threaded n_context to the train split only would leave val at width 1
    and silently degenerate the seed to an IC broadcast.
    """
    n_ctx = 4
    arr = np.random.randn(8, 13, 16, 16).astype(np.float32)
    path = tmp_path / "syn.npy"
    np.save(path, arr)

    dm = KFDataModule(data_path=str(path), n_train=4, n_val=2,
                      batch_size=1, sub_t=2, n_context=n_ctx)
    dm.setup(stage="fit")

    assert dm.val_dataset[0]["ctx"].shape[-1] == n_ctx
    assert dm.train_dataset[0]["ctx"].shape[-1] == n_ctx
