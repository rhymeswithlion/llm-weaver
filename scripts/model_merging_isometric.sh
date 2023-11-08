#!/usr/bin/env bash

# Import common things
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $SCRIPTS_DIR/common.sh
cd $SCRIPTS_DIR/../model_merging

#######################
# Run the isometric model merging as shown in ./model_merging/README.md

EVAL_TASK=rte
RTE_MODEL=textattack/roberta-base-RTE
MNLI_MODEL=textattack/roberta-base-MNLI
# Isometric merge.
python3 ./scripts/merge_and_evaluate.py  \
    --models=$RTE_MODEL,$MNLI_MODEL \
    --glue_task=$EVAL_TASK
