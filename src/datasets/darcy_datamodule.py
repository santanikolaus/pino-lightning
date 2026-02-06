from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import pytorch_lightning as L
from torch.utils.data import DataLoader

#TODO implement this locally
from legacy.neuralop.data.datasets import load_darcy_flow_small
from legacy.neuralop.data.transforms.data_processors import DataProcessor


class DarcyDataModule(L.LightningDataModule):
    """Thin wrapper that exposes the legacy Darcy loaders to Lightning."""

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

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_loader is not None and self._test_loaders is not None:
            return

        load_kwargs = dict(
            n_train=self.n_train,
            n_tests=list(self.n_tests),
            batch_size=self.batch_size,
            test_batch_sizes=list(self.test_batch_sizes),
            test_resolutions=list(self.test_resolutions),
            encode_input=self.encode_input,
            encode_output=self.encode_output,
            encoding=self.encoding,
            channel_dim=self.channel_dim,
        )
        if self.data_root is not None:
            load_kwargs["data_root"] = self.data_root

        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            **load_kwargs
        )
        self._train_loader = train_loader
        self._test_loaders = test_loaders
        self.data_processor = data_processor

    @property
    def train_loader(self) -> DataLoader:
        if self._train_loader is None:
            raise RuntimeError("setup must be called before accessing the train loader")
        return self._train_loader

    @property
    def test_loaders(self) -> Dict[int, DataLoader]:
        if self._test_loaders is None:
            raise RuntimeError("setup must be called before accessing test loaders")
        return self._test_loaders

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self):
        return [self.test_loaders[res] for res in self.test_resolutions]

    def test_dataloader(self):
        return [self.test_loaders[res] for res in self.test_resolutions]

    def legacy_loaders(self) -> Tuple[DataLoader, Dict[int, DataLoader], Optional[DataProcessor]]:
        """Return the tuple produced by ``load_darcy_flow_small`` for debugging."""
        return self.train_loader, self.test_loaders, self.data_processor
