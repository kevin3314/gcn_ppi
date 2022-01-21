import logging
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

from .mixin import LabelMixin, NumFeatureMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NumDataset(Dataset, NumFeatureMixin, LabelMixin):
    def __init__(self, csv_path: Union[str, Path], tsv_path: Union[str, Path]):
        self.load_numerical_features(csv_path, tsv_path)
        self.load_label(csv_path)

    def __getitem__(self, index):
        return (
            self.numerical_features0[index],
            self.numerical_features1[index],
            self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.numerical_features0)
