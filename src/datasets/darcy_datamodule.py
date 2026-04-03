from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.datasets.darcy_dataset import DarcyDataset
from src.datasets.transforms.data_processors import DefaultDataProcessor


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
        input_coord_channels: bool = False,
        sparse_input_resolution: Optional[int] = None,
        smooth_a_sigma: Optional[float] = None,
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
        self.input_coord_channels = input_coord_channels
        self.sparse_input_resolution = sparse_input_resolution
        self.smooth_a_sigma = smooth_a_sigma

        # Stride divisibility checks
        if (self.source_resolution - 1) % (self.train_resolution - 1) != 0:
            raise ValueError(
                f"source_resolution ({self.source_resolution}) and train_resolution "
                f"({self.train_resolution}) not on same vertex grid: "
                f"({self.source_resolution}-1) % ({self.train_resolution}-1) != 0"
            )
        if self.sparse_input_resolution is not None:
            if (self.source_resolution - 1) % (self.sparse_input_resolution - 1) != 0:
                raise ValueError(
                    f"source_resolution ({self.source_resolution}) and sparse_input_resolution "
                    f"({self.sparse_input_resolution}) not on same vertex grid: "
                    f"({self.source_resolution}-1) % ({self.sparse_input_resolution}-1) != 0"
                )

        if len(self.n_tests) != len(self.test_resolutions):
            raise ValueError("n_tests must have the same length as test_resolutions")
        if len(self.test_batch_sizes) != len(self.test_resolutions):
            raise ValueError("test_batch_sizes must have the same length as test_resolutions")

        self._train_loader: Optional[DataLoader] = None
        self._test_loaders: Optional[Dict[int, DataLoader]] = None
        self.data_processor: Optional[DefaultDataProcessor] = None

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
            input_coord_channels=self.input_coord_channels,
        )
        if self.data_root is not None:
            load_kwargs["root_dir"] = self.data_root
        if self.sparse_input_resolution is not None:
            load_kwargs["sparse_input_resolution"] = self.sparse_input_resolution
        if self.smooth_a_sigma is not None:
            load_kwargs["smooth_a_sigma"] = self.smooth_a_sigma

        dataset = DarcyDataset(**load_kwargs)

        self._train_loader = DataLoader(
            dataset.train_db,
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
