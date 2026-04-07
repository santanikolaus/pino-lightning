import numpy as np
import torch
from torch.utils.data import Dataset


class KFDataset(Dataset):
    """Kolmogorov Flow dataset.

    Loads a single NS_fine_Re{re}_T{t_res}_part{n}.npy file.
    Raw shape on disk: (N, T+1, S, S) float32.
    Stored in memory as: (n_samples, S, S, T_eff) — channels-last, time is last dim,
    where T_eff = (T+1 - 1) // sub_t + 1 effective frames after temporal subsampling.

    Args:
        path: path to .npy file
        n_samples: number of trajectories to load
        offset: starting trajectory index (default 0)
        sub_t: temporal subsampling stride (default 1 = no subsampling).
               sub_t=2 matches the paper's Table 8 setup: load T=128 files and
               subsample every 2nd frame → 65 effective frames at dt=1/64.
    """

    def __init__(self, path: str, n_samples: int, offset: int = 0, *, sub_t: int):
        raw = np.load(path, mmap_mode='r')
        # Slice the requested window
        chunk = raw[offset: offset + n_samples]                  # (n_samples, T+1, S, S)
        # Temporal subsampling along axis 1 (time), matching paper: data[..., ::sub_t, ...]
        if sub_t > 1:
            chunk = chunk[:, ::sub_t, :, :]                      # (n_samples, T_eff, S, S)
        # Permute (N, T_eff, S, S) → (N, S, S, T_eff) and copy into RAM
        arr = np.ascontiguousarray(chunk.transpose(0, 2, 3, 1))
        self.data = torch.from_numpy(arr)                        # float32 preserved

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict:
        traj = self.data[idx]  # (S, S, T+1)
        ic = traj[..., 0]      # (S, S) — first time frame
        return {"x": ic, "y": traj}
