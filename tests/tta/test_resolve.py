from msc.tta import setup

_EXPECTED_NROW6J9L = {
    "data": {
        "data_path": "/system/user/studentwork/wehofer/data/ns/NS_fine_Re500_T128_res128_part0.npy",
        "n_train": 240,
        "n_val": 30,
        "batch_size": 1,
        "num_workers": 0,
        "offset_train": 0,
        "sub_t": 2,
        "temporal_pad": 5,
        "pad_mode": "zero",
        "time_scale": 1.0,
        "coarse_path": None,
        "coarse_shuffle_p": None,
        "coarse_paths": None,
    },
    "model": {
        "model_arch": "fno",
        "data_channels": 4,
        "out_channels": 1,
        "n_modes": [8, 8, 8],
        "hidden_channels": 128,
        "n_layers": 4,
        "lifting_channel_ratio": 0,
        "projection_channel_ratio": 2,
        "domain_padding": 0.0,
        "positional_embedding": None,
        "norm": None,
        "fno_skip": "linear",
        "implementation": "factorized",
        "use_channel_mlp": False,
        "channel_mlp_expansion": 0.5,
        "channel_mlp_dropout": 0.0,
        "separable": False,
        "factorization": None,
        "rank": 1.0,
        "fixed_rank_modes": False,
        "stabilizer": "None",
    },
    "loss": {
        "re": 500,
        "t_interval": 1.0,
        "data_weight": 5.0,
        "pde_weight": 1.0,
        "ic_weight": 1.0,
    },
    "trainer": {
        "_target_": "lightning.Trainer",
        "max_epochs": 150,
        "accelerator": "gpu",
        "devices": 1,
        "precision": 32,
        "enable_model_summary": False,
        "check_val_every_n_epoch": 1,
        "limit_train_batches": None,
        "limit_val_batches": None,
        "limit_test_batches": None,
    },
    "logger": {
        "wandb": {
            "_target_": "lightning.pytorch.loggers.WandbLogger",
            "project": "msc-base",
            "name": "pino-re500-pretrain",
        }
    },
    "callbacks": {
        "model_checkpoint": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "monitor": "val_l2",
            "mode": "min",
            "save_top_k": 1,
            "save_last": False,
            "filename": "best",
        },
        "kf_visualizer": {
            "_target_": "src.callbacks.kf_visualizer.KFVisualizerCallback",
            "log_every_n_epochs": 50,
        },
    },
    "opt": {
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "milestones": [25, 50, 75, 100],
        "step_size": 100,
        "gamma": 0.5,
    },
}


def test_resolve_matches_known_run():
    """Locks resolve()'s full output for a known run — guards silent wandb/Hydra drift."""
    cfg = setup.resolve("nrow6j9l")
    assert cfg == _EXPECTED_NROW6J9L
