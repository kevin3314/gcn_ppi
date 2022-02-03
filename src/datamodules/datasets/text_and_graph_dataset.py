from pathlib import Path
from typing import Dict, Union

import torch
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .mixin import GraphDataMixin, LabelMixin, TextMixin


class TextAndGraphDataset(Dataset, TextMixin, GraphDataMixin, LabelMixin):
    def __init__(
        self,
        csv_path: Union[str, Path],
        pdbid2node: Dict[str, torch.Tensor],
        pdbid2adj: Dict[str, coo_matrix],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256,
    ):
        self.load_text(csv_path, tokenizer, max_seq_len=max_seq_len)
        self.load_label(csv_path)
        self.load_pdb_data(csv_path, pdbid2node, pdbid2adj)

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.token_type_ids[index],
            self.attention_mask[index],
            self.amino_acids_graph_list0[index],
            self.amino_acids_graph_list1[index],
            self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.input_ids)
