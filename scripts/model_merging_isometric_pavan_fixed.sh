#!/usr/bin/env bash

# Start at the repo root
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR/..

# Activate virtual environment
source source_to_activate.sh

# Install dependencies
pip install -q absl-py tensorflow tensorflow_datasets tensorflow_probability


#######################
# Run the isometric model merging as shown in ./model_merging/README.md

cd $THIS_DIR/../model_merging
export PYTHONPATH=.

EVAL_TASK=rte
# RTE_MODEL=textattack/roberta-base-RTE
# MNLI_MODEL=textattack/roberta-base-MNLI

RTE_MODEL=~/ucb-devenv/2023-fall-cs-294-merging-llms/gpt2-finetuned-rte
MNLI_MODEL=~/ucb-devenv/2023-fall-cs-294-merging-llms/gpt2-finetuned-mnli

# This still has null for the pad token
# RTE_MODEL=PavanNeerudu/gpt2-finetuned-rte
# MNLI_MODEL=PavanNeerudu/gpt2-finetuned-mnli

# Isometric merge.
PYTHONPATH=. python3 ./scripts/merge_and_evaluate.py  \
    --models=$RTE_MODEL,$MNLI_MODEL \
    --glue_task=$EVAL_TASK
