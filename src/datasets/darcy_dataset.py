import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

from src.datasets.pt_datasets import PTDataset, vertex_stride
from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer

logger = logging.getLogger(__name__)


def _build_coord_grid(resolution: int) -> torch.Tensor:
    """Vertex-centered [x, y] coordinate grid including both endpoints 0 and 1.

    Returns shape (2, resolution, resolution).
    """
    coords = torch.linspace(0, 1, resolution)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=0)


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
        input_coord_channels: bool = False,
        sparse_input_resolution: Optional[int] = None,
        smooth_a_sigma: Optional[float] = None,
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
        self._input_coord_channels = input_coord_channels
        self._sparse_input_resolution = sparse_input_resolution
        self._smooth_a_sigma = smooth_a_sigma
        self._train_resolution = train_resolution  # needed in _process_test

        # Validate strides before any I/O
        vertex_stride(source_resolution, train_resolution)
        for res in test_resolutions:
            vertex_stride(source_resolution, res)
        if sparse_input_resolution is not None:
            vertex_stride(source_resolution, sparse_input_resolution)

        super().__init__(
            root_dir=root_dir,
            dataset_name="darcy",
            source_resolution=source_resolution,
            n_train=n_train,
            n_tests=n_tests,
            train_resolution=train_resolution,
            test_resolutions=list(test_resolutions),
        )

    def _smooth_x(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to the permeability channel (channel 0) of x.

        x shape: (N, C, H, W) where C=1 (permeability only, before coord concat).
        sigma applied to spatial dims only; batch and channel dims get sigma=0.
        """
        if self._smooth_a_sigma is None:
            return x
        x_np = x.numpy()
        x_np = gaussian_filter(x_np, sigma=(0, 0, self._smooth_a_sigma, self._smooth_a_sigma))
        return torch.from_numpy(x_np)

    def _process_train(self, data: dict, n_train: int, resolution: int):
        if self._sparse_input_resolution is not None:
            # Load x at sparse resolution, NN-fill to train_resolution (value replication,
            # no blending — binary {3,12} preserved everywhere)
            sparse_stride = vertex_stride(self._source_resolution, self._sparse_input_resolution)
            x_train = data["x"].type(torch.float32).clone().unsqueeze(self._channel_dim)
            x_train = x_train[:n_train, :, ::sparse_stride, ::sparse_stride]  # (N,1,11,11)
            x_train = F.interpolate(x_train, size=(resolution, resolution), mode='nearest')  # (N,1,61,61)
            x_train = self._smooth_x(x_train)
            # Labels stay at sparse resolution
            y_train = data["y"].clone().unsqueeze(self._channel_dim)
            y_train = y_train[:n_train, :, ::sparse_stride, ::sparse_stride]  # (N,1,11,11)
        else:
            stride = vertex_stride(self._source_resolution, resolution)
            x_train = data["x"].type(torch.float32).clone()
            x_train = x_train.unsqueeze(self._channel_dim)
            x_train = x_train[:n_train, :, ::stride, ::stride]
            x_train = self._smooth_x(x_train)
            y_train = data["y"].clone()
            y_train = y_train.unsqueeze(self._channel_dim)
            y_train = y_train[:n_train, :, ::stride, ::stride]

        if self._input_coord_channels:
            grid = _build_coord_grid(resolution)  # (2, H, W) — always at train_resolution
            grid = grid.unsqueeze(0).expand(x_train.size(0), -1, -1, -1)
            x_train = torch.cat([x_train, grid], dim=1)  # (N, 3, H, W)

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

        if self._sparse_input_resolution is not None and resolution == self._sparse_input_resolution:
            # For the sparse-resolution test set: NN-fill x to train_resolution, keep y at sparse res.
            # This matches the training distribution (model always sees 61×61 NN-filled inputs).
            sparse_stride = vertex_stride(self._source_resolution, self._sparse_input_resolution)
            x_test = data["x"].type(torch.float32).clone().unsqueeze(self._channel_dim)
            x_test = x_test[:n_test, :, ::sparse_stride, ::sparse_stride]
            x_test = F.interpolate(x_test, size=(self._train_resolution, self._train_resolution), mode='nearest')
            x_test = self._smooth_x(x_test)
            y_test = data["y"].clone().unsqueeze(self._channel_dim)
            y_test = y_test[:n_test, :, ::sparse_stride, ::sparse_stride]
            coord_res = self._train_resolution
        else:
            stride = vertex_stride(self._source_resolution, resolution)
            x_test = data["x"].type(torch.float32).clone()
            x_test = x_test.unsqueeze(self._channel_dim)
            x_test = x_test[:n_test, :, ::stride, ::stride]
            x_test = self._smooth_x(x_test)
            y_test = data["y"].clone()
            y_test = y_test.unsqueeze(self._channel_dim)
            y_test = y_test[:n_test, :, ::stride, ::stride]
            coord_res = resolution

        if self._input_coord_channels:
            grid = _build_coord_grid(coord_res)
            grid = grid.unsqueeze(0).expand(x_test.size(0), -1, -1, -1)
            x_test = torch.cat([x_test, grid], dim=1)

        return TensorDataset(x_test, y_test)
