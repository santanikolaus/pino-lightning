import logging
from pathlib import Path
from typing import List, Union, Optional

from src.datasets.pt_datasets import PTDataset
from src.datasets.web_utils import download_from_zenodo_record

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
        download: bool = True,
        pde_resolution: Optional[int] = None,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        zenodo_record_id = "12784353"

        # Validate stride divisibility for all target resolutions
        all_resolutions = set(test_resolutions + [train_resolution])
        if pde_resolution is not None:
            all_resolutions.add(pde_resolution)
        for res in all_resolutions:
            if (source_resolution - 1) % (res - 1) != 0:
                raise ValueError(
                    f"Cannot stride-subsample source_resolution={source_resolution} "
                    f"to target resolution={res}: "
                    f"({source_resolution}-1) % ({res}-1) != 0"
                )

        if download:
            files_to_download = []
            already_downloaded_files = [x.name for x in root_dir.iterdir()]
            if (
                f"darcy_train_{source_resolution}.pt" not in already_downloaded_files
                or f"darcy_test_{source_resolution}.pt" not in already_downloaded_files
            ):
                files_to_download.append(f"darcy_{source_resolution}.tgz")
            download_from_zenodo_record(
                record_id=zenodo_record_id,
                root=root_dir,
                files_to_download=files_to_download,
            )

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


def load_darcy(
    n_train,
    n_tests,
    data_root,
    test_resolutions=[16, 32],
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    train_resolution: int = 16,
    source_resolution: int = 421,
    download: bool = True,
    pde_resolution: Optional[int] = None,
):

    return DarcyDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        channel_dim=channel_dim,
        encoding=encoding,
        source_resolution=source_resolution,
        download=download,
        pde_resolution=pde_resolution,
    )
