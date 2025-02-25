from typing import Callable, Tuple
import h5py
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset

from .h5 import HDF5Dataset


class PairedDataset(Dataset):
    def __init__(self, root: str, transform: Callable = None):
        super(PairedDataset, self).__init__()

        self.root = root
        self.transform = transform

        self.h5_ds = HDF5Dataset(self.root)

    def __len__(self):
        return len(self.h5_ds)

    def __getitem__(self, index: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        shard_idx, idx_in_shard = self.h5_ds.get_indices(index)

        with h5py.File(self.h5_ds.shard_paths[shard_idx], "r") as f:
            pair = f[str(idx_in_shard)]

            source = pair["source"][()]
            target = pair["target"][()]

        source = Tensor(source)
        if self.transform is not None:
            target = np.transpose(target, (1, 2, 0)).astype(np.uint8)
            target = self.transform(target)
        else:
            target = Tensor(target)

        return (source, target), 0
