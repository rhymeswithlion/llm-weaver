#!/usr/bin/env bash

# Import common things
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $SCRIPTS_DIR/common.sh
cd $SCRIPTS_DIR/../model_merging

#######################
# Run the fischer model merging as shown in ./model_merging/README.md

cd $THIS_DIR/../model_merging
export PYTHONPATH=.

EVAL_TASK=rte
# original
# RTE_MODEL=textattack/roberta-base-RTE
# MNLI_MODEL=textattack/roberta-base-MNLI
# FISHER_DIR=./fisher_coeffs_roberta_rte_mnli

# doesn't work. no tokenizer pad token
# RTE_MODEL=PavanNeerudu/gpt2-finetuned-rte
# MNLI_MODEL=PavanNeerudu/gpt2-finetuned-mnli
# FISHER_DIR=./fisher_coeffs_pavan_gpt2_rte_mnli

RTE_MODEL=~/ucb-devenv/2023-fall-cs-294-merging-llms/gpt2-finetuned-rte
MNLI_MODEL=~/ucb-devenv/2023-fall-cs-294-merging-llms/gpt2-finetuned-mnli
FISHER_DIR=./fisher_coeffs_pavan_gpt2_rte_mnli



# RTE_MODEL=George-Ogden/gpt2-finetuned-mnli
# MNLI_MODEL=George-Ogden/gpt2-finetuned-mnli
# FISHER_DIR=./fisher_coeffs_ogden_gpt2_rte_mnli

mkdir -p $FISHER_DIR


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
