import logging
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset

from src.datasets.pt_datasets import PTDataset, vertex_stride
from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer

logger = logging.getLogger(__name__)


def _reduce_dims(ndim: int, encoding: str, channel_dim: int) -> List[int]:
    if encoding == "channel-wise":
        dims = list(range(ndim))
        dims.pop(channel_dim)
        return dims
    elif encoding == "pixel-wise":
        return [0]
    else:
        raise ValueError(f"Unknown encoding {encoding!r}")


class TensorDataset(Dataset):

    def __init__(self, x, y):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self):
        return self.x.size(0)


class DarcyDataset(PTDataset):

    def __init__(
        self,
        root_dir: Union[Path, str],
        n_train: int,
        n_tests: List[int],
        train_resolution: int,
        test_resolutions: List[int] = [16, 32],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding: str = "channel-wise",
        channel_dim: int = 1,
        source_resolution: int = 421,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        # Store before super().__init__(), which immediately calls _process_train/_process_test
        self._encode_input = encode_input
        self._encode_output = encode_output
        self._encoding = encoding
        self._channel_dim = channel_dim
        self._source_resolution = source_resolution

        # Validate strides before any I/O
        vertex_stride(source_resolution, train_resolution)
        for res in test_resolutions:
            vertex_stride(source_resolution, res)

        super().__init__(
            root_dir=root_dir,
            dataset_name="darcy",
            source_resolution=source_resolution,
            n_train=n_train,
            n_tests=n_tests,
            train_resolution=train_resolution,
            test_resolutions=list(test_resolutions),
        )

    def _process_train(self, data: dict, n_train: int, resolution: int):
        stride = vertex_stride(self._source_resolution, resolution)

        x_train = data["x"].type(torch.float32).clone()
        x_train = x_train.unsqueeze(self._channel_dim)
        x_train = x_train[:n_train, :, ::stride, ::stride]

        y_train = data["y"].clone()
        y_train = y_train.unsqueeze(self._channel_dim)
        y_train = y_train[:n_train, :, ::stride, ::stride]

        if self._encode_input:
            input_encoder = UnitGaussianNormalizer(dim=_reduce_dims(
                x_train.ndim, self._encoding, self._channel_dim))
            input_encoder.fit(x_train)
        else:
            input_encoder = None

        if self._encode_output:
            output_encoder = UnitGaussianNormalizer(dim=_reduce_dims(
                y_train.ndim, self._encoding, self._channel_dim))
            output_encoder.fit(y_train)
        else:
            output_encoder = None

        return (
            TensorDataset(x_train, y_train),
            DefaultDataProcessor(in_normalizer=input_encoder,
                                 out_normalizer=output_encoder),
        )

    def _process_test(self, data: dict, n_test: int,
                      resolution: int) -> Dataset:
        logger.info("Loading test db for resolution %s with %s samples",
                    resolution, n_test)
        stride = vertex_stride(self._source_resolution, resolution)

        x_test = data["x"].type(torch.float32).clone()
        x_test = x_test.unsqueeze(self._channel_dim)
        x_test = x_test[:n_test, :, ::stride, ::stride]

        y_test = data["y"].clone()
        y_test = y_test.unsqueeze(self._channel_dim)
        y_test = y_test[:n_test, :, ::stride, ::stride]

        return TensorDataset(x_test, y_test)
