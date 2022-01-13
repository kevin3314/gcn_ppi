<div align="center">

# Protein-protein Interaction Prediction Utilizing Multi-modal Information

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
Predict protein-protein interaction (PPI) utilizing multi-modality, including text, molecular structure (*Graph*), and numerical feature.
Transformer-based models and graph neural networks are dedicated for text and graph, respectively.\
Core idea comes from [Multimodal Graph-based Transformer Framework for Biomedical Relation Extraction](https://aclanthology.org/2021.findings-acl.328/).\
Our's differs in that our model for protein structural modality process over residues rather than atoms.

![Overview](imgs/model.png)



## Results
We list the Precision/Recall/F1 scores of each models in the following table.

| Model | HPRD50 | BioInfer |
| :--- | ---: |---: |
| Dutta et al. [1] Text | 90.44/58.67/71.17 | 54.42/87.45/67.09 |
| Dutta et al. [1] Text & Graph| 94.79/75.21/83.87 | 69.04/88.49/77.54 |
| Pingali et al. [2] &dagger; | 95.47/94.69/95.06 | 78.49/79.78/80.86 |
| Text only | 75.0/85.7/80.0 | 74.0/64.7/69.0 |
| Text + Protein graph | **100.0**/**85.7**/**92.3** | TBW |
| Text + Protein graph + Numerical feature | 80.0/60.0/68.6 | TBW |

&dagger;: The evaluation metrics in the author's implementation seem broken, though. Their text modality model is too simple yet has beaten previous models, including strong pretrained model-based, Bio-BERT-based one. Moreover, we found bugs in their implementation of metrics.

> [1]: Pratik Dutta and Sriparna Saha, Amalgamation of protein sequence, structure and textual information for improving protein-protein interaction identification, In Proceedings of the 58th Annual Meet- ing of the Association for Computational Linguistics\
> [2] Sriram Pingali, Shweta Yadav, Pratik Dutta and Sriparna Saha, Multimodal Graph-based Transformer Framework for Biomedical Relation Extraction, Findings of the Association for Computational Linguistics: ACL-IJCNLP

## Requirements
Dependency is maintained by poetry. Some dependencies (ones related to pytorch-geometric), however, can not be installed via poetry and need to be installed manually.
Please follow [instructions](https://github.com/pyg-team/pytorch_geometric#installation).
We used libraries compatible with pytorch 1.10.0.
```console
$ CUDA=cu102  # cpu, cu102, or cu113
$ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
$ pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
$ pip install torch-geometric
```

## Preprocess
1. Download PPI data annotated with gene names from [here](https://github.com/duttaprat/MM_PPI_NLP) to data/mm_data.
2. Convert xlsm file to csv file (suppose `HPRD50_multimodal_dataset.csv`).
3. List up gene names in `HPRD50_multimodal_dataset.csv` by `preprocess/list_gene_names.py`.
4. Fetch pdb ids and ensemble ids corresponding to gene names by `preprocess/convert_to_pdb_ensemble_id.py`.
5. Fetch pdb files corresponding to pdb ids by `preprocess/fetch_pdb_by_id.py`.
6. Complement pdb id by `preprocess/complement_pdb_id.py`.
7. Split csv by `preprocess/split_csv.py`
8. Preprocess on pdb files (e.g. building adjacency matrix) by `preprocess.py`.
    - Set configuration on `configs/preprocess.yaml`.
9. (Optional) Normalize numerical feature by `preprocess/normalize_numerical.py`

```console
$ python preprocess/list_gene_names.py data/mm_data/HPRD50_multimodal_dataset.csv  data/mm_data/hprd50_gene_name.txt
$ python preprocess/convert_to_pdb_ensemble_id.py data/mm_data/hprd50_gene_name.txt data/mm_data/genename2emsembl_pdb.json
$ python preprocess/fetch_pdb_by_id.py data/mm_data/genename2emsembl_pdb.json data/pdb
$ python preprocess/complement_pdb_id.py data/mm_data/HPRD50_multimodal_dataset.csv data/mm_data/HPRD50_multimodal_dataset_with_pdb.csv data/mm_data/genename2emsebl_pdb.json
$ python preprocess/split_csv.py data/mm_data/HPRD50_multimodal_dataset.csv 0.8,0.1,0.1 data/hprd50
$ python preprocess_pdb.py
$ python preprocess/normalize_numerical.py data/emsemble2feature/gene_numerical.tsv data/emsemble2feature/normalized_gene_numerical.tsv  # Optional
```

## How to run

Train model with default configuration.
If needed, you may modify configuration in `configs/hprd50_config.yaml` and its dependents.
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

Test with trained model.
```yaml
python test.py load_checkpoint=path/to/checkpoint
```

## Hyper Parameters
Hyper parameters are listed in model configuration file as well. For more detail, you may refer to it.

| Option | Values|
| :--- | ---: |
| Optimizer | AdamW |
| batch size | 32 |
| Maximum epochs | 20 |
| Learning scheduler | Linear scheduler |
| Learning rate | 5e-5 |
| Warmup Epoch | 5 |
| Weight Decay | 0.01 |
| Node dimension of GNN | 128 |
| The number of GNN layers | 2 |
| Dimension of numerical feature | 180 |

<br>
