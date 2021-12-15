<div align="center">

# Protein-protein Interaction Prediction by GNN

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
Predict protein-protein interaction (PPI) by GNN tailored to text modality and protein structural modality.\
This repo's core idea is similar to [Multimodal Graph-based Transformer Framework for Biomedical Relation Extraction](https://aclanthology.org/2021.findings-acl.328/).\
Main differences is that our model for protein structural modality process over residues rather than atoms.



## Results
We list only the result of the HPRD50 dataset because the BioInfer dataset, annotated with gene names, distributed [here](https://github.com/duttaprat/MM_PPI_NLP) is broken.

| Model | HPRD50 |
| :--- | ---: |
| Text only | 60.8 |
| Text + Protein graph | 92.3 |

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

```console
$ python preprocess/list_gene_names.py data/mm_data/HPRD50_multimodal_dataset.csv  data/mm_data/hprd50_gene_name.txt
$ python preprocess/convert_to_pdb_ensemble_id.py data/mm_data/hprd50_gene_name.txt data/mm_data/genename2emsembl_pdb.json
$ python preprocess/fetch_pdb_by_id.py data/mm_data/genename2emsembl_pdb.json data/pdb
$ python preprocess/complement_pdb_id.py data/mm_data/HPRD50_multimodal_dataset.csv data/mm_data/HPRD50_multimodal_dataset_with_pdb.csv data/mm_data/genename2emsebl_pdb.json
$ python preprocess/split_csv.py data/mm_data/HPRD50_multimodal_dataset.csv 0.8,0.1,0.1 data/hprd50
$ python preprocess_pdb.py
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

<br>
