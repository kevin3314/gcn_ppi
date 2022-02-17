#!/bin/bash -eu
text=false
graph=false
numerical=fales
cross_validation=false
dataset=""

while getopts d:tgnc opt; do
    case "$opt" in
        c)
            cross_validation=true
            ;;
        t)
            text=true
            ;;
        g)
            graph=true
            ;;
        n)
            numerical=true
            ;;
        d)
            dataset="$OPTARG"
            ;;
        -)
            break
            ;;
        \?)
            exit 1
            ;;
        -*)
            echo "$0: illegal option -- ${opt##-}" >&2
            exit 1
            ;;
    esac
done
if [ "$dataset" = "hprd" ]; then
    train_size=2586
    data="$(pwd)/data/hprd50/all.csv"
elif [ "$dataset" = "bioinfer" ]; then
    train_size=13676
    data="$(pwd)/data/bioinfer/all.csv"
else
    echo "Dataset ${dataset} not found"
    echo "Available datasets: hprd, bioinfer"
    exit 1
fi

echo "dataset -> $dataset"
echo "text -> $text"
echo "graph-> $graph"
echo "numerical-> $numerical"

BASE_CMD="python run.py trainer=ddp trainer.gpus=2 experiment_name=$dataset model.train_size=$train_size datamodule.csv_path=$data do_cross_validation=$cross_validation"
BASE_MODEL="_model"
BASE_DATASET="_dataset"

if [ $text = false ] && [ $graph = false ] && [ $numerical = false ]; then
    continue
fi

# Construct base args.
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

if [ $numerical = true ]; then
    if [ "$MODEL_ARG" != "" ]; then
        MODEL_ARG="${MODEL_ARG}_and_num"
    else
        MODEL_ARG="num"
    fi

    if [ "$DATA_ARG" != "" ]; then
        DATA_ARG="${DATA_ARG}_num"
    else
        DATA_ARG="num"
    fi
fi

MODEL_ARG="${MODEL_ARG}${BASE_MODEL}"
DATA_ARG="${DATA_ARG}${BASE_DATASET}"
CMD="${BASE_CMD} model=$MODEL_ARG datamodule=$DATA_ARG"
# If numerical is true, iterate all numerical files.
if [ $numerical = true ]; then
    # Feature version
    for version in 1 2
    do
        if [ $version = 1 ]; then
            dims=(100 180)
        elif [ $version = 2 ]; then
            dims=(100 300 500)
        fi

        for dim in "${dims[@]}"
        do
            num_feature_path="$(pwd)/data/emsemble2feature/gene_feature_v${version}_log_pca${dim}.tsv"
            RUN_CMD="${CMD} datamodule.feature_tsv_path=$num_feature_path model.num_feature_dim=$dim"
            if [ $text = true ]; then
                CMD_WITH_LOW_TENSORFUSION="${RUN_CMD} model.with_lowrank_tensorfusion_network=true"
                echo "run: $CMD_WITH_LOW_TENSORFUSION"
                eval $CMD_WITH_LOW_TENSORFUSION

                CMD_WITH_TENSORFUSION="${RUN_CMD} model.with_tensorfusion_network=true"
                echo "run: $CMD_WITH_TENSORFUSION"
                eval $CMD_WITH_TENSORFUSION

                CMD_WITH_INTERMEDIATE="${RUN_CMD} model.with_intermediate_layer=true"
                echo "run: $CMD_WITH_INTERMEDIATE"
                eval $CMD_WITH_INTERMEDIATE

                CMD_PLAIN="${RUN_CMD}"
                echo "run: $CMD_PLAIN"
                eval $CMD_PLAIN
            else
                echo "run: $RUN_CMD"
                eval $RUN_CMD
            fi
        done
    done
else
    echo "run: $CMD"
    eval $CMD
fi
