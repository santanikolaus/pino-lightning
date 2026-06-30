import random

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional


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
        coarse_path: optional path to a band-limited coarse trajectory file (same shape
                     as path). When provided, items include a "coarse" key with the
                     low-passed trajectory (S, S, T_eff). Must be derived from the same
                     source as path so that index alignment holds.
        coarse_shuffle_p: probability of replacing the matched coarse trajectory with a
                          random other sample's coarse during __getitem__. Only set on
                          the training dataset; leave at 0.0 for val/test.
        coarse_ic_only: when True, batch["coarse"] is the IC frame (t=0) of the coarse
                        trajectory broadcast across all T time steps instead of the full
                        trajectory. Use for phase experiments where the 5th FNO channel
                        should carry only IC-time amplitude information.
        coarse_paths: list of paths to sibling coarse trajectory files (same shape as path).
                      Each file provides one sibling channel; batch["coarse"] is
                      (n_sibs, S, S, T_eff). Mutually exclusive with coarse_path.
    """

    def __init__(self, path: str, n_samples: int, offset: int = 0, *,
                 sub_t: int, coarse_path: Optional[str] = None,
                 coarse_shuffle_p: float = 0.0,
                 coarse_ic_only: bool = False,
                 coarse_paths: Optional[List[str]] = None,
                 n_context: int = 1):
        raw = np.load(path, mmap_mode='r')
        # Slice the requested window
        chunk = raw[offset: offset + n_samples]                  # (n_samples, T+1, S, S)
        # Temporal subsampling along axis 1 (time), matching paper: data[..., ::sub_t, ...]
        if sub_t > 1:
            chunk = chunk[:, ::sub_t, :, :]                      # (n_samples, T_eff, S, S)
        # Permute (N, T_eff, S, S) → (N, S, S, T_eff) and copy into RAM
        arr = np.ascontiguousarray(chunk.transpose(0, 2, 3, 1))
        self.data = torch.from_numpy(arr)                        # float32 preserved

        self.coarse = None
        if coarse_path is not None:
            raw_c = np.load(coarse_path, mmap_mode='r')
            chunk_c = raw_c[offset: offset + n_samples]
            if sub_t > 1:
                chunk_c = chunk_c[:, ::sub_t, :, :]
            self.coarse = torch.from_numpy(
                np.ascontiguousarray(chunk_c.transpose(0, 2, 3, 1))
            )
        self.coarse_shuffle_p = coarse_shuffle_p
        self.coarse_ic_only = coarse_ic_only

        self.n_context = n_context

        self.coarses = None
        if coarse_paths is not None:
            self.coarses = []
            for cp in coarse_paths:
                raw_c = np.load(cp, mmap_mode='r')
                chunk_c = raw_c[offset: offset + n_samples]
                if sub_t > 1:
                    chunk_c = chunk_c[:, ::sub_t, :, :]
                self.coarses.append(torch.from_numpy(
                    np.ascontiguousarray(chunk_c.transpose(0, 2, 3, 1))
                ))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict:
        traj = self.data[idx]  # (S, S, T+1)
        ic = traj[..., 0]      # (S, S) — first time frame
        out = {"x": ic, "y": traj, "ctx": traj[..., :self.n_context]}
        if self.coarses is not None:
            out["coarse"] = torch.stack([c[idx] for c in self.coarses], dim=0)
        elif self.coarse is not None:
            coarse_idx = idx
            if self.coarse_shuffle_p > 0.0 and random.random() < self.coarse_shuffle_p:
                coarse_idx = random.randint(0, len(self) - 1)
                if coarse_idx == idx and len(self) > 1:
                    coarse_idx = (idx + 1) % len(self)
            if self.coarse_ic_only:
                raw_ic = self.coarse[coarse_idx][..., 0]          # (S, S)
                out["coarse"] = raw_ic.unsqueeze(-1).expand(-1, -1, traj.shape[-1]).clone()
            else:
                out["coarse"] = self.coarse[coarse_idx]
        return out

