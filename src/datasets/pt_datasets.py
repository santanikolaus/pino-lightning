from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset

from src.datasets.transforms.data_processors import DefaultDataProcessor


def vertex_stride(source: int, target: int) -> int:
    """Compute stride for vertex-centered subsampling.

    Formula: stride = (source - 1) // (target - 1), valid only when
    (source - 1) % (target - 1) == 0.
    """
    if (source - 1) % (target - 1) != 0:
        raise ValueError(f"Cannot stride-subsample {source} -> {target}: "
                         f"({source}-1) % ({target}-1) != 0")
    return (source - 1) // (target - 1)


class PTDataset(ABC):
    """Abstract base for datasets stored as .pt files.

    Owns file loading (naming convention: {dataset_name}_{split}_{source_resolution}.pt)
    and exposes train_db / test_dbs / data_processor properties.

    Subclasses implement _process_train and _process_test to handle
    PDE-specific data layout (spatial striding, time axes, normalizers, etc.).
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        dataset_name: str,
        source_resolution: int,
        n_train: int,
        n_tests: List[int],
        train_resolution: int,
        test_resolutions: List[int],
    ):
        root_dir = Path(root_dir)
        train_data = torch.load(
            root_dir / f"{dataset_name}_train_{source_resolution}.pt",
            weights_only=False,
        )
        test_data = torch.load(
            root_dir / f"{dataset_name}_test_{source_resolution}.pt",
            weights_only=False,
        )

        self._train_db, self._data_processor = self._process_train(
            train_data, n_train, train_resolution)
        self._test_dbs = {
            res: self._process_test(test_data, n_test, res)
            for res, n_test in zip(test_resolutions, n_tests)
        }
        del train_data, test_data

    @abstractmethod
    def _process_train(self, data: dict, n_train: int,
                       resolution: int) -> Tuple[Dataset, DefaultDataProcessor]:
        """Return (train_db, data_processor) for the given training resolution."""
        ...

    @abstractmethod
    def _process_test(self, data: dict, n_test: int,
                      resolution: int) -> Dataset:
        """Return a Dataset for the given test resolution."""
        ...

    @property
    def data_processor(self) -> DefaultDataProcessor:
        return self._data_processor

    @property
    def train_db(self) -> Dataset:
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
