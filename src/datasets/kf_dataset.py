import numpy as np
import torch
from torch.utils.data import Dataset


class KFDataset(Dataset):
    """Kolmogorov Flow dataset.

    Loads a single NS_fine_Re{re}_T{t_res}_part{n}.npy file.
    Raw shape on disk: (N, T+1, S, S) float32.
    Stored as: (n_samples, S, S, T+1) — channels-last, time is last dim.

    Args:
        path: path to .npy file
        n_samples: number of trajectories to load
        offset: starting trajectory index (default 0)
    """

    def __init__(self, path: str, n_samples: int, offset: int = 0):
        raw = np.load(path, mmap_mode='r')
        # Slice the requested window, then permute (N,T+1,S,S) → (N,S,S,T+1)
        chunk = raw[offset: offset + n_samples]                  # (n_samples, T+1, S, S)
        arr = np.ascontiguousarray(chunk.transpose(0, 2, 3, 1))  # copy into RAM
        self.data = torch.from_numpy(arr)                        # float32 preserved

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict:
        traj = self.data[idx]  # (S, S, T+1)
        ic = traj[..., 0]      # (S, S) — first time frame
        return {"x": ic, "y": traj}
