from pathlib import Path
from typing import Union

from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .mixin import GraphDataMixin, LabelMixin, TextMixin


class TextAndGraphDataset(Dataset, TextMixin, GraphDataMixin, LabelMixin):
    def __init__(
        self,
        csv_path: Union[str, Path],
        pdb_processed_root: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256,
    ):
        self.load_text(csv_path, tokenizer, max_seq_len=max_seq_len)
        self.load_label(csv_path)
        self.load_pdb_data(csv_path, pdb_processed_root)

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.token_type_ids[index],
            self.attention_mask[index],
            self.amino_acids_graph_list0[index],
            self.amino_acids_graph_list1[index],
            self.labels[index],
        )
