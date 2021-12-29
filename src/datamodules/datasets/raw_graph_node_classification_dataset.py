import torch
from torch.utils.data import Dataset


class RawGraphNodeClassificationDataset(Dataset):
    def __init__(
        self,
        raw_embeddings: torch.Tensor,
        wl_embeddings: torch.Tensor,
        hop_embeddings: torch.Tensor,
        int_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.raw_embeddings = raw_embeddings
        self.wl_embeddings = wl_embeddings
        self.hop_embeddings = hop_embeddings
        self.int_embeddings = int_embeddings
        self.labels = labels

    def __getitem__(self, index):
        raw_features = self.raw_embeddings[index]
        wl_features = self.wl_embeddings[index]
        hop_features = self.hop_embeddings[index]
        int_features = self.int_embeddings[index]
        label = self.labels[index]
        return (
            raw_features,
            torch.ones(1),
            torch.ones(1),
            wl_features,
            int_features,
            hop_features,
            label,
        )

    def __len__(self):
        return len(self.raw_embeddings)
