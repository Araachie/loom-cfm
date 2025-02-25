from typing import Callable, Tuple
import h5py

from torch import Tensor
from torch.utils.data import Dataset

from .h5 import HDF5Dataset


class H5Dataset(Dataset):
    def __init__(self, root: str, transform: Callable = None):
        super(H5Dataset, self).__init__()

        self.root = root
        self.transform = transform

        self.h5_ds = HDF5Dataset(self.root)

    def __len__(self):
        return len(self.h5_ds)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        shard_idx, idx_in_shard = self.h5_ds.get_indices(index)

        with h5py.File(self.h5_ds.shard_paths[shard_idx], "r") as f:
            handle = f[str(idx_in_shard)]
            sample = handle["image"][()]

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample = Tensor(sample)

        return sample, 0
