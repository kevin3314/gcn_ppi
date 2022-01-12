import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, load_npz
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

NULL_EMBEDDING = torch.ones(1)
NULL_ADJ = coo_matrix((0, 0))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LabelMixin:
    def load_label(self, csv_path: Union[str, Path]) -> torch.Tensor:
        logger.info("Loading label from {}".format(csv_path))
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        labels = df["GOLD"].astype(np.float32)
        labels = torch.from_numpy(np.array(labels))
        self.labels = labels


class TextMixin:
    def load_text(self, csv_path: Union[str, Path], tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 256):
        logger.info("Loading text data from {}".format(csv_path))
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        texts = df["text"].values
        pad_token_id = tokenizer.pad_token_id

        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        exceeded_samples = 0

        # Tokenize
        for text in texts:
            inputs = tokenizer(text)
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            if len(input_ids) > max_seq_len:
                exceeded_samples += 1
                input_ids = input_ids[:max_seq_len]
                token_type_ids = token_type_ids[:max_seq_len]
                attention_mask = attention_mask[:max_seq_len]
            while len(input_ids) < max_seq_len:
                input_ids.append(pad_token_id)
                token_type_ids.append(0)
                attention_mask.append(0)
            self.input_ids.append(torch.Tensor(input_ids).long())
            self.token_type_ids.append(torch.Tensor(token_type_ids).long())
            self.attention_mask.append(torch.Tensor(attention_mask).long())

        logger.info(f"Found {exceeded_samples} samples exceeding max sequence length of {max_seq_len} in {csv_path}")


class NumFeatureMixin:
    def load_numerical_features(self, csv_path: Union[str, Path], tsv_path: Union[str, Path]):
        """Load numerical features from given tsv file.
        Suppose the csv file has the following format:
        ENSEMBL:ENSG00000242268 0.0     0.0     0.0     0.0     0.0     0
        ENSEMBL:ENSG00000270112 0.0     124.973306483   0.0     0.0     0.0

        Args:
            tsv_path (Union[str, Path]): Path to feature tsv file.
        """
        logger.info("Loading numerical feature from {}".format(tsv_path))
        numerical_df = pd.read_csv(tsv_path, delimiter="\t", header=None)
        numerical_df[0] = numerical_df[0].map(lambda x: x.replace("ENSEMBL:", ""))
        emsembl_id2features = {}
        for _, row in numerical_df.iterrows():
            emsembl_id2features[row[0]] = torch.from_numpy(row[1:].values.astype(np.float32))
        # Dummy feature in case of missing features
        NULL_FEATURE = torch.zeros_like(emsembl_id2features[list(emsembl_id2features.keys())[0]])

        df = pd.read_csv(csv_path)
        emsembl_ids0 = df["ENSEMBLE_ID0"].values
        emsembl_ids1 = df["ENSEMBLE_ID1"].values
        numerical_features0: List[torch.Tensor] = [
            emsembl_id2features.get(emsembl_id, NULL_FEATURE) for emsembl_id in emsembl_ids0
        ]
        numerical_features1: List[torch.Tensor] = [
            emsembl_id2features.get(emsembl_id, NULL_FEATURE) for emsembl_id in emsembl_ids1
        ]

        missing_num = sum([(x == NULL_FEATURE).all() for x in numerical_features0]) + sum(
            [(x == NULL_FEATURE).all() for x in numerical_features1]
        )
        logger.info(
            "# of genes missing numerical feature: {} out of {}".format(
                missing_num, len(numerical_features0) + len(numerical_features1)
            )
        )
        self.numerical_features0 = numerical_features0
        self.numerical_features1 = numerical_features1


class GraphDataMixin:
    @staticmethod
    def get_pdb_nodes(pdb_ids0: List[str], pdb_ids1: List[str], pdb_root: Path) -> Tuple[List[torch.Tensor], ...]:
        res0 = [
            torch.from_numpy(np.load(pdb_root / f"{pdb_id}_ids.npy"))
            if (pdb_root / f"{pdb_id}_ids.npy").exists()
            else NULL_EMBEDDING
            for pdb_id in pdb_ids0
        ]
        res1 = [
            torch.from_numpy(np.load(pdb_root / f"{pdb_id}_ids.npy"))
            if (pdb_root / f"{pdb_id}_ids.npy").exists()
            else NULL_EMBEDDING
            for pdb_id in pdb_ids1
        ]

        missing_num = sum([(x == NULL_EMBEDDING).all() for x in res0]) + sum(
            [(x == NULL_EMBEDDING).all() for x in res1]
        )
        return res0, res1, missing_num

    @staticmethod
    def get_adj_matrix(pdb_ids0: List[str], pdb_ids1: List[str], pdb_root: Path) -> Tuple[List[coo_matrix], ...]:
        res0 = [
            load_npz(pdb_root / f"{pdb_id}_adj.npz") if (pdb_root / f"{pdb_id}_adj.npz").exists() else NULL_ADJ
            for pdb_id in pdb_ids0
        ]
        res1 = [
            load_npz(pdb_root / f"{pdb_id}_adj.npz") if (pdb_root / f"{pdb_id}_adj.npz").exists() else NULL_ADJ
            for pdb_id in pdb_ids1
        ]

        missing_num = sum([(x == NULL_ADJ).toarray().all() for x in res0]) + sum(
            [(x == NULL_ADJ).toarray().all() for x in res1]
        )
        return res0, res1, missing_num
        return res0, res1, missing_num

    def load_pdb_data(self, csv_path: Union[str, Path], pdb_processed_root: Union[str, Path]) -> Tuple[List[Data], ...]:
        logger.info("Loading graph feature from {}".format(pdb_processed_root))
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        amino_acids_list0, amino_acids_list1, missing_nodes = GraphDataMixin.get_pdb_nodes(
            df["PDB_ID0"].values, df["PDB_ID1"], pdb_processed_root
        )
        amino_acids_adj_list0, amino_acids_adj_list1, missing_adjs = GraphDataMixin.get_adj_matrix(
            df["PDB_ID0"].values, df["PDB_ID1"], pdb_processed_root
        )
        amino_acids_edges0: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list0]
        amino_acids_edges1: List[torch.Tensor] = [from_scipy_sparse_matrix(adj)[0] for adj in amino_acids_adj_list1]

        logger.info(
            "# of missing nodes: {} out of {}".format(missing_nodes, len(amino_acids_list0) + len(amino_acids_list1))
        )
        logger.info(
            "# of missing adjs: {} out of {}".format(
                missing_adjs, len(amino_acids_adj_list0) + len(amino_acids_adj_list1)
            )
        )

        amino_acids_graph_list0 = [
            Data(amino_acids_node, amino_acids_edge)
            for amino_acids_node, amino_acids_edge in zip(amino_acids_list0, amino_acids_edges0)
        ]
        amino_acids_graph_list1 = [
            Data(amino_acids_node, amino_acids_edge)
            for amino_acids_node, amino_acids_edge in zip(amino_acids_list1, amino_acids_edges1)
        ]
        self.amino_acids_graph_list0 = amino_acids_graph_list0
        self.amino_acids_graph_list1 = amino_acids_graph_list1
