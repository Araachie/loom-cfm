import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        super(IndexedDataset, self).__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[index]


class IndexedDatasetWithFlips(Dataset):
    def __init__(self, dataset: Dataset, flip_p: float):
        super(IndexedDatasetWithFlips, self).__init__()

        self.dataset = dataset
        self.flip_p = flip_p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index][0]
        if torch.rand([1]).item() < self.flip_p:
            sample = T.RandomHorizontalFlip(p=1.0)(sample)
            index = index + len(self.dataset)
        return index, sample
