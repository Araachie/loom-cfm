import os
from typing import Callable, Optional, Tuple, Union
from zipfile import ZipFile

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


class ZipDataset(Dataset):
    def __init__(self, root: str, transform: Callable = None):
        super(ZipDataset, self).__init__()

        with open(os.path.join(root, "samples.txt"), 'r') as f:
            self.samples = sorted(
                [row.strip() for row in f.readlines()]
            )

        self.root = root

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[int]]:
        path = self.samples[index]

        with ZipFile(os.path.join(self.root, "data.zip"), 'r') as zip_file:
            with zip_file.open(path) as f:
                sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, None
