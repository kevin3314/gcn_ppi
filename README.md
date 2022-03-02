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

![Overview](imgs/overview.png)



## Results
We list the Precision/Recall/F1 scores of each models in the following table.

Followings are results of **single** run.
| Model | HPRD50 | BioInfer |
| :--- | ---: |---: |
| Dutta et al. [1] (Text) | 90.44/58.67/71.17 | 54.42/87.45/67.09 |
| Dutta et al. [1] (Text & Graph) | 94.79/75.21/83.87 | 69.04/88.49/77.54 |
| Pingali et al. [2] &dagger; | **95.47**/**94.69**/**95.06** | 78.49/79.78/80.86 |
| Text only | 93.3/70.0/80.0 | 81.2/86.4/83.7 |
| Graph only | 0.00/0.00/0.00 | 0.00/0.00/0.00 |
| Numerical only | 9.68/30.0/14.6 | 13.1/16.6/14.7 |
| Text + Graph | 87.5/70.0/77.8 | **84.4**/85.1/84.8 |
| Text + Numerical | 79.0/75.0/76.9 | 80.5/**88.1**/84.2 |
| Graph + Numerical | 0.00/0.00/0.00 | 19.3/29.4/23.3 |
| Text + Graph + Numerical | 82.4/70.0/75.7 | 83.8/**88.1**/**85.9** |

&dagger;: The evaluation metrics in the author's implementation seem broken, though. Their text modality model is too simple yet has beaten previous models, including strong pretrained model-based, Bio-BERT-based one. Moreover, we found bugs in their implementation of metrics.

> [1]: Pratik Dutta and Sriparna Saha, Amalgamation of protein sequence, structure and textual information for improving protein-protein interaction identification, In Proceedings of the 58th Annual Meet- ing of the Association for Computational Linguistics\
> [2] Sriram Pingali, Shweta Yadav, Pratik Dutta and Sriparna Saha, Multimodal Graph-based Transformer Framework for Biomedical Relation Extraction, Findings of the Association for Computational Linguistics: ACL-IJCNLP

Followings are results of **cross** validation.
Results on HPRD50

| Model | val/f1 | test/acc | test/prec | test/rec | test/f1 | test/auroc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 78.14 (±4.80) | 97.60 (±1.37) | 80.92 (±18.47) | 70.85 (±16.59) | 73.48 (±12.15) | 94.45 (±8.05) |
| Graph | 10.60 (±5.43) | 67.05 (±20.25) | 2.57 (±2.28) | 27.13 (±22.83) | 4.68 (±4.14) | 51.37 (±7.86) |
| Num | 18.07 (±8.53) | 69.69 (±19.87) | 7.67 (±2.70) | 37.82 (±18.00) | 11.45 (±1.98) | 55.81 (±9.57) |
| Text + Graph | 74.68 (±4.99) | 97.36 (±0.75) | 75.39 (±10.89) | 67.78 (±19.24) | 68.99 (±9.81) | 92.31 (±7.11) |
| Text + Num | **78.47** (±7.98) | 97.60 (±0.99) | 80.30 (±15.33) | 69.33 (±16.20) | 72.18 (±11.11) | **96.24** (±5.15) |
| Graph + Num | 18.32 (±6.81) | 68.68 (±9.15) | 4.00 (±2.66) | 30.43 (±21.32) | 7.06 (±4.72) | 48.13 (±6.93) |
| Text + Graph + Num (Concat) | 78.07 (±7.36) | **98.29** (±0.94) | **89.17** (±1.36) | **72.85** (±16.96) | **79.25** (±10.78) | 95.37 (±5.73) |

Results on Bioinfer

| Model | val/f1 | test/acc | test/prec | test/rec | test/f1 | test/auroc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text | 85.85 (±2.00) | 94.66 (±1.08) | 84.73 (±3.75) | 85.69 (±3.32) | 85.14 (±2.71) | **97.31** (±0.85) |
| Graph | 1.48 (±2.25) | 81.70 (±1.23) | 5.71 (±11.43) | 1.26 (±2.53) | 2.07 (±4.14) | 51.06 (±1.07) |
| Num | 17.24 (±4.52) | 70.27 (±2.37) | 16.79 (±2.66) | 17.43 (±4.85) | 16.99 (±3.85) | 50.94 (±3.63) |
| Text + Graph | 84.41 (±2.08) | 94.22 (±1.20) | 85.58 (±3.46) | 81.58 (±6.32) | 83.37 (±3.58) | 96.24 (±1.66) |
| Text + Num | **86.54** (±2.92) | 94.72 (±1.13) | 85.49 (±3.51) | **84.73** (±4.09) | 85.06 (±3.27) | 96.57 (±1.31) |
| Graph + Num | 21.81 (±1.20) | 63.75 (±0.92) | 16.43 (±1.16) | 25.61 (±4.68) | 19.94 (±2.13) | 49.84 (±1.16) |
| Text + Graph + Num (Concat) | 86.48 (±3.49) | **94.82** (±1.04) | **86.93** (±2.92) | **83.63** (±2.99) | **85.23** (±2.74) | 96.22 (±1.52) |

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
3. List up gene names in `HPRD50_multimodal_dataset.csv` by `preprocess/list_gene_names.py`. (You need specify dataset name)
4. Fetch pdb ids and ensemble ids corresponding to gene names by `preprocess/fetch_pdb_ensemble_id.py`.
5. Fetch pdb files corresponding to pdb ids by `preprocess/fetch_pdb_by_id.py`.
6. Complement pdb id by `preprocess/complement_pdb_id.py`.

```console
$ python preprocess/list_gene_names.py data/mm_data/HPRD50_multimodal_dataset.csv  data/mm_data/hprd50_gene_name.txt
$ python preprocess/fetch_pdb_ensemble_id.py data/mm_data/hprd50_gene_name.txt data/mm_data/genename2emsembl_pdb.json [hprd/bioinfer]
$ python preprocess/fetch_pdb_by_id.py data/mm_data/genename2emsembl_pdb.json data/pdb
$ python preprocess/complement_pdb_id.py data/mm_data/HPRD50_multimodal_dataset.csv data/[hprd/bioinfer]/all.csv data/mm_data/genename2emsebl_pdb.json
```

NOTE: Resultant csv file should be located at **data/[hprd/bioinfer]/all.csv**

The PDB files are translated into graphs on the fly.
The result will be cached in the directory specified by *CACHE_ROOT* environment variable because processing takes a little time.
You may set it by .env file (see .env.example).


## How to run

The method is evaluated based on 5-fold cross validation (you may change the number of folds).
If needed, you may modify configuration in `configs/train.yaml` and its dependents.
```console
# default
$ python run.py

# train on CPU
$ python run.py trainer.gpus=0

# train on GPU
$ python run.py trainer.gpus=1
```

You may run all configure combination by make command.
```console
# Run all combination
$ make all DATASET=hprd

# Run text model
$ make text DATASET=hprd
```
You can override any parameter from command line like this
```console
$ python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

All results are maintained by mlflow. You can launch mlflow server by `mlflow ui`.
```console
$ mlflow ui
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
