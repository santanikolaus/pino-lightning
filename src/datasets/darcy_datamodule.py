from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.datasets.darcy_dataset import load_darcy
from src.datasets.transforms.data_processors import DataProcessor


class PairedResolutionDataset(Dataset):
    """Wraps a base dataset, adding a high-resolution permeability to each sample.

    Used for the PINO native forward pass: the FNO evaluates at ``pde_resolution``
    using the native high-res coefficient ``a``, while the data loss compares
    against labels at the original (lower) resolution.
    """

    def __init__(self, base: Dataset, a_highres: torch.Tensor) -> None:
        self.base = base
        self.a_highres = a_highres

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["a_highres"] = self.a_highres[idx]
        return sample


class DarcyDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        n_train: int,
        n_tests: Sequence[int],
        batch_size: int,
        test_batch_sizes: Sequence[int],
        data_root: Optional[Union[Path, str]] = None,
        test_resolutions: Sequence[int] = (16, 32),
        encode_input: bool = False,
        encode_output: bool = True,
        encoding: str = "channel-wise",
        channel_dim: int = 1,
        train_resolution: int = 16,
        source_resolution: int = 421,
        download: bool = False,
        pde_resolution: Optional[int] = None,

    ) -> None:
        super().__init__()
        self.n_train = n_train
        self.n_tests = tuple(n_tests)
        self.batch_size = batch_size
        self.test_batch_sizes = tuple(test_batch_sizes)
        self.data_root = Path(data_root).expanduser() if data_root is not None else None
        self.test_resolutions = tuple(test_resolutions)
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.encoding = encoding
        self.channel_dim = channel_dim
        self.train_resolution = train_resolution
        self.source_resolution = source_resolution
        self.download = download
        self.pde_resolution = pde_resolution

        # Stride divisibility checks
        if (self.source_resolution - 1) % (self.train_resolution - 1) != 0:
            raise ValueError(
                f"source_resolution ({self.source_resolution}) and train_resolution "
                f"({self.train_resolution}) not on same vertex grid: "
                f"({self.source_resolution}-1) % ({self.train_resolution}-1) != 0"
            )
        if self.pde_resolution is not None:
            if (self.source_resolution - 1) % (self.pde_resolution - 1) != 0:
                raise ValueError(
                    f"source_resolution ({self.source_resolution}) and pde_resolution "
                    f"({self.pde_resolution}) not on same vertex grid: "
                    f"({self.source_resolution}-1) % ({self.pde_resolution}-1) != 0"
                )
            if (self.pde_resolution - 1) % (self.train_resolution - 1) != 0:
                raise ValueError(
                    f"pde_resolution ({self.pde_resolution}) and train_resolution "
                    f"({self.train_resolution}) not on same vertex grid: "
                    f"({self.pde_resolution}-1) % ({self.train_resolution}-1) != 0"
                )

        if len(self.n_tests) != len(self.test_resolutions):
            raise ValueError(
                "n_tests must have the same length as test_resolutions"
            )
        if len(self.test_batch_sizes) != len(self.test_resolutions):
            raise ValueError(
                "test_batch_sizes must have the same length as test_resolutions"
            )

        self._train_loader: Optional[DataLoader] = None
        self._test_loaders: Optional[Dict[int, DataLoader]] = None
        self.data_processor: Optional[DataProcessor] = None

    #TODO: reintroduce MultigridPatching2D, MGPatchingDataProcessor once we port the legacy helper
    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_loader is not None and self._test_loaders is not None:
            return

        load_kwargs = dict(
            n_train=self.n_train,
            n_tests=list(self.n_tests),
            test_resolutions=list(self.test_resolutions),
            encode_input=self.encode_input,
            encode_output=self.encode_output,
            encoding=self.encoding,
            channel_dim=self.channel_dim,
            train_resolution=self.train_resolution,
            source_resolution=self.source_resolution,
            download=self.download,
        )
        if self.pde_resolution is not None:
            load_kwargs["pde_resolution"] = self.pde_resolution
        if self.data_root is not None:
            load_kwargs["data_root"] = self.data_root

        dataset = load_darcy(**load_kwargs)

        train_db: Dataset = dataset.train_db

        # When pde_resolution differs from train_resolution, load native
        # high-res permeability (stride from source) and pair it with each
        # training sample.
        if (self.pde_resolution is not None
                and self.pde_resolution != self.train_resolution
                and self.data_root is not None):
            source_path = (
                Path(self.data_root).expanduser()
                / f"darcy_train_{self.source_resolution}.pt"
            )
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source Darcy training data not found at '{source_path}'. "
                    f"source_resolution={self.source_resolution} requires this file. "
                    f"Set download=True or ensure the file is present."
                )
            source_data = torch.load(source_path.as_posix(), weights_only=False)
            a_full = source_data["x"].type(torch.float32).clone()[:self.n_train]
            del source_data

            pde_stride = (self.source_resolution - 1) // (self.pde_resolution - 1)
            a_highres = a_full[:, ::pde_stride, ::pde_stride]
            a_highres = a_highres.unsqueeze(self.channel_dim)  # (N, 1, H, W)
            del a_full
            train_db = PairedResolutionDataset(train_db, a_highres)

        self._train_loader = DataLoader(
            train_db,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        self._test_loaders = {
            res: DataLoader(
                dataset.test_dbs[res],
                batch_size=bs,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )
            for res, bs in zip(self.test_resolutions, self.test_batch_sizes)
        }
        self.data_processor = dataset.data_processor

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self):
        return [self._test_loaders[res] for res in self.test_resolutions]

    def test_dataloader(self):
        return [self._test_loaders[res] for res in self.test_resolutions]
