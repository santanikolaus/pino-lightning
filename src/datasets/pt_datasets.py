from pathlib import Path
from typing import List, Union
import torch
import logging

from src.datasets.tensor_dataset import TensorDataset
from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer

logger = logging.getLogger(__name__)

class PTDataset:

    def __init__(
        self,
        root_dir: Union[Path, str],
        dataset_name: str,
        n_train: int,
        n_tests: List[int],
        train_resolution: int,
        test_resolutions: List[int],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding="channel-wise",
        input_subsampling_rate=None,
        output_subsampling_rate=None,
        channel_dim=1,
        channels_squeezed=True,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.test_resolutions = test_resolutions

        data = torch.load(
        Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}.pt").as_posix()
        )

        x_train = data["x"].type(torch.float32).clone()
        if channels_squeezed:
            x_train = x_train.unsqueeze(channel_dim)

        input_data_dims = x_train.ndim - 2
        if not input_subsampling_rate:
            input_subsampling_rate = 1
        if not isinstance(input_subsampling_rate, list):
            input_subsampling_rate = [input_subsampling_rate] * input_data_dims
        assert len(input_subsampling_rate) == input_data_dims, (
            "Error: length mismatch between input_subsampling_rate and data dims. "
            f"Expected {input_data_dims}, got {input_subsampling_rate}"
        )
        train_input_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in input_subsampling_rate]
        train_input_indices.insert(channel_dim, slice(None))
        train_input_indices = tuple(train_input_indices)
        x_train = x_train[train_input_indices]

        y_train = data["y"].clone()
        if channels_squeezed:
            y_train = y_train.unsqueeze(channel_dim)

        # TODO(patching): If a dataset ever passes channels_squeezed=False, switch
        # this indexing logic to build full slices explicitly instead of inserting
        # channel_dim—otherwise tensors that already have C in place can misalign.

        output_data_dims = y_train.ndim - 2
        if not output_subsampling_rate:
            output_subsampling_rate = 1
        if not isinstance(output_subsampling_rate, list):
            output_subsampling_rate = [output_subsampling_rate] * output_data_dims
        assert len(output_subsampling_rate) == output_data_dims, (  # NEW
            "Error: length mismatch between output_subsampling_rate and data dims. "
            f"Expected {output_data_dims}, got {output_subsampling_rate}"
        )

        train_output_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in output_subsampling_rate]
        train_output_indices.insert(channel_dim, slice(None))
        train_output_indices = tuple(train_output_indices)
        y_train = y_train[train_output_indices]

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

        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            logger.info("Loading test db for resolution %s with %s samples", res, n_test)
            data = torch.load(Path(root_dir).joinpath(f"{dataset_name}_test_{res}.pt").as_posix())

            x_test = data["x"].type(torch.float32).clone()
            if channels_squeezed:
                x_test = x_test.unsqueeze(channel_dim)
            test_input_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in input_subsampling_rate]
            test_input_indices.insert(channel_dim, slice(None))
            test_input_indices = tuple(test_input_indices)
            x_test = x_test[test_input_indices]

            y_test = data["y"].clone()
            if channels_squeezed:
                y_test = y_test.unsqueeze(channel_dim)
            test_output_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in output_subsampling_rate]
            test_output_indices.insert(channel_dim, slice(None))
            test_output_indices = tuple(test_output_indices)
            y_test = y_test[test_output_indices]

            del data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs
