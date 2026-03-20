from pathlib import Path
from typing import List, Union
import torch
import logging

from src.datasets.tensor_dataset import TensorDataset
from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer

logger = logging.getLogger(__name__)


def _vertex_stride(source: int, target: int) -> int:
    """Compute stride for vertex-centered subsampling.

    Formula: stride = (source - 1) // (target - 1), valid only when
    (source - 1) % (target - 1) == 0.
    """
    if (source - 1) % (target - 1) != 0:
        raise ValueError(
            f"Cannot stride-subsample {source} -> {target}: "
            f"({source}-1) % ({target}-1) != 0"
        )
    return (source - 1) // (target - 1)


class PTDataset:

    def __init__(
        self,
        root_dir: Union[Path, str],
        dataset_name: str,
        n_train: int,
        n_tests: List[int],
        train_resolution: int,
        test_resolutions: List[int],
        source_resolution: int,
        encode_input: bool = False,
        encode_output: bool = True,
        encoding="channel-wise",
        channel_dim=1,
        channels_squeezed=True,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.test_resolutions = test_resolutions

        # ── Train data: load source, stride to train_resolution ──────────
        train_stride = _vertex_stride(source_resolution, train_resolution)
        data = torch.load(
            Path(root_dir).joinpath(f"{dataset_name}_train_{source_resolution}.pt").as_posix()
        )

        x_train = data["x"].type(torch.float32).clone()
        if channels_squeezed:
            x_train = x_train.unsqueeze(channel_dim)
        x_train = x_train[:n_train, :, ::train_stride, ::train_stride]

        y_train = data["y"].clone()
        if channels_squeezed:
            y_train = y_train.unsqueeze(channel_dim)
        y_train = y_train[:n_train, :, ::train_stride, ::train_stride]

        del data

        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None

        self._train_db = TensorDataset(
            x_train,
            y_train,
        )

        self._data_processor = DefaultDataProcessor(
            in_normalizer=input_encoder, out_normalizer=output_encoder
        )

        # ── Test data: load source once, stride to each test resolution ──
        self._test_dbs = {}
        test_data = torch.load(
            Path(root_dir).joinpath(f"{dataset_name}_test_{source_resolution}.pt").as_posix()
        )
        for res, n_test in zip(test_resolutions, n_tests):
            logger.info("Loading test db for resolution %s with %s samples", res, n_test)
            test_stride = _vertex_stride(source_resolution, res)

            x_test = test_data["x"].type(torch.float32).clone()
            if channels_squeezed:
                x_test = x_test.unsqueeze(channel_dim)
            x_test = x_test[:n_test, :, ::test_stride, ::test_stride]

            y_test = test_data["y"].clone()
            if channels_squeezed:
                y_test = y_test.unsqueeze(channel_dim)
            y_test = y_test[:n_test, :, ::test_stride, ::test_stride]

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

        del test_data

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
