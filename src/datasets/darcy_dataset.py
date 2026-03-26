import logging
from pathlib import Path
from typing import List, Union

from src.datasets.pt_datasets import PTDataset


logger = logging.getLogger(__name__)


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
        encoding="channel-wise",
        channel_dim=1,
        source_resolution: int = 421,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        super().__init__(
            root_dir=root_dir,
            dataset_name="darcy",
            n_train=n_train,
            n_tests=n_tests,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            source_resolution=source_resolution,
            encode_input=encode_input,
            encode_output=encode_output,
            encoding=encoding,
            channel_dim=channel_dim,
        )
