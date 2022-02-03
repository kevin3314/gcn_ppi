#!/bin/bash -eu

dataset="$1"

if [ "$dataset" = "hprd" ]; then
    train_size=2586
    data="$(pwd)/data/hprd50/all.csv"
elif [ "$dataset" = "bioinfer" ]; then
    train_size=13676
    data="$(pwd)/data/bioinfer/all.csv"
else
    echo "Dataset not found"
    echo "Available datasets: hprd, bioinfer"
    exit 1
fi

BASE_CMD="python run.py trainer=ddp trainer.gpus=2 experiment_name=$dataset model.train_size=$train_size datamodule.csv_path=$data"
BASE_MODEL="_model"
BASE_DATASET="_dataset"

#FIXME: This does not handle numerical models.
for text in true false
do
    for graph in true false
    do
        if [ $text = false ] && [ $graph = false ]; then
            continue
        fi
        echo "text -> $text"
        echo "graph-> $graph"

        MODEL_ARG=""
        DATA_ARG=""
        if [ $text = true ]; then
            MODEL_ARG="text"
            DATA_ARG="text"
        fi

        if [ $graph = true ]; then
            if [ "$MODEL_ARG" != "" ]; then
                MODEL_ARG="${MODEL_ARG}_and_graph"
            else
                MODEL_ARG="graph"
            fi

            if [ "$DATA_ARG" != "" ]; then
                DATA_ARG="${DATA_ARG}_graph"
            else
                DATA_ARG="graph"
            fi
        fi

        MODEL_ARG="${MODEL_ARG}${BASE_MODEL}"
        DATA_ARG="${DATA_ARG}${BASE_DATASET}"
        CMD="${BASE_CMD} model=$MODEL_ARG datamodule=$DATA_ARG"
        echo "run: $CMD"
        eval $CMD
    done
done
