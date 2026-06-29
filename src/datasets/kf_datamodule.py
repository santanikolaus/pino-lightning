from typing import List, Optional

import lightning as L
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset


class KFDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_path: str,
        n_train: int,
        n_val: int,
        batch_size: int = 2,
        num_workers: int = 0,
        offset_train: int = 0,
        offset_val: Optional[int] = None,
        *,
        sub_t: int,
        coarse_path: Optional[str] = None,
        coarse_shuffle_p: float = 0.0,
        coarse_ic_only: bool = False,
        coarse_paths: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.offset_train = offset_train
        self.offset_val = offset_val if offset_val is not None else offset_train + n_train
        self.sub_t = sub_t
        self.coarse_path = coarse_path
        self.coarse_shuffle_p = coarse_shuffle_p
        self.coarse_ic_only = coarse_ic_only
        self.coarse_paths = coarse_paths

    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(self, "train_dataset"):
            return
        self.train_dataset = KFDataset(self.data_path, self.n_train, offset=self.offset_train,
                                       sub_t=self.sub_t, coarse_path=self.coarse_path,
                                       coarse_shuffle_p=self.coarse_shuffle_p,
                                       coarse_ic_only=self.coarse_ic_only,
                                       coarse_paths=self.coarse_paths)
        self.val_dataset = KFDataset(self.data_path, self.n_val, offset=self.offset_val,
                                     sub_t=self.sub_t, coarse_path=self.coarse_path,
                                     coarse_ic_only=self.coarse_ic_only,
                                     coarse_paths=self.coarse_paths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
