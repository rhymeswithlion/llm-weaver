#!/usr/bin/env bash

set -e

# Start at the repo root
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR/..
pwd

# Activate virtual environment
source source_to_activate.sh

# Install dependencies
pip install absl-py tensorflow tensorflow_datasets tensorflow_probability


#######################
# Run the fischer model merging as shown in ./model_merging/README.md

cd $THIS_DIR/../model_merging
export PYTHONPATH=.
mkdir -p fisher_coefficients

EVAL_TASK=rte
RTE_MODEL=textattack/roberta-base-RTE
MNLI_MODEL=textattack/roberta-base-MNLI
FISHER_DIR=./fisher_coefficients

# Compute RTE Fisher.
python3 ./scripts/compute_fisher.py  \
    --model=$RTE_MODEL \
    --glue_task="rte" \
    --fisher_path="$FISHER_DIR/rte_fisher.h5"

# Compute MNLI Fisher.
python3 ./scripts/compute_fisher.py  \
    --model=$MNLI_MODEL \
    --glue_task="mnli" \
    --fisher_path="$FISHER_DIR/mnli_fisher.h5"

# Fisher merge
python3 ./scripts/merge_and_evaluate.py  \
    --models=$RTE_MODEL,$MNLI_MODEL \
    --fishers=$FISHER_DIR/rte_fisher.h5,$FISHER_DIR/mnli_fisher.h5 \
    --glue_task=$EVAL_TASK
