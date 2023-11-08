#!/usr/bin/env bash

# Import common things
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $SCRIPTS_DIR/common.sh
cd $SCRIPTS_DIR/../model_merging

#######################

EVAL_TASK=rte


RTE_MODEL=George-Ogden/gpt2-finetuned-mnli
MNLI_MODEL=George-Ogden/gpt2-finetuned-mnli
FISHER_DIR=./fisher_coeffs_ogden_gpt2_rte_mnli

mkdir -p $FISHER_DIR


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

# $ ./scripts/model_merging_fisher.sh
# Using Python from /Users/briancruz/2023-fall-cs-194-294-merging-llms/scripts/../.venv/bin/python
# model_merging package found: /Users/briancruz/2023-fall-cs-194-294-merging-llms/model_merging/model_merging/__init__.py
# All PyTorch model weights were used when initializing TFGPT2ForSequenceClassification.

# All the weights of TFGPT2ForSequenceClassification were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2ForSequenceClassification for predictions without further training.
# I1108 14:54:03.046181 8280416384 compute_fisher.py:38] Model loaded
# I1108 14:54:03.050359 8280416384 dataset_info.py:578] Load dataset info from /Users/briancruz/tensorflow_datasets/glue/rte/2.0.0
# I1108 14:54:03.053048 8280416384 dataset_builder.py:528] Reusing dataset glue (/Users/briancruz/tensorflow_datasets/glue/rte/2.0.0)
# I1108 14:54:03.079472 8280416384 logging_logger.py:49] Constructing tf.data.Dataset glue for split train, from /Users/briancruz/tensorflow_datasets/glue/rte/2.0.0
# /Users/briancruz/2023-fall-cs-194-294-merging-llms/.venv/lib/python3.8/site-packages/transformers/data/processors/glue.py:520: FutureWarning: This processor will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
#   warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)
# I1108 14:54:03.142910 8280416384 compute_fisher.py:47] Dataset loaded
# I1108 14:54:03.142972 8280416384 compute_fisher.py:49] Starting Fisher computation
# I1108 14:57:31.921828 8280416384 compute_fisher.py:52] Fisher computed. Saving to file...
# I1108 14:57:31.924141 8280416384 compute_fisher.py:55] Fisher saved to file
# All PyTorch model weights were used when initializing TFGPT2ForSequenceClassification.

# All the weights of TFGPT2ForSequenceClassification were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2ForSequenceClassification for predictions without further training.
# I1108 14:57:39.049520 8280416384 compute_fisher.py:38] Model loaded
# 2023-11-08 14:57:39.106305: W tensorflow/tsl/platform/cloud/google_auth_provider.cc:184] All attempts to get a Google authentication bearer token failed, returning an empty token. Retrieving token from files failed with "NOT_FOUND: Could not locate the credentials file.". Retrieving token from GCE failed with "FAILED_PRECONDITION: Error executing an HTTP request: libcurl code 6 meaning 'Couldn't resolve host name', error details: Could not resolve host: metadata.google.internal".
# I1108 14:57:39.649979 8280416384 dataset_info.py:736] Load pre-computed DatasetInfo (eg: splits, num examples,...) from GCS: glue/mnli/2.0.0
# I1108 14:57:40.643023 8280416384 dataset_info.py:578] Load dataset info from /var/folders/5s/r3p544z57kxbmgkzvz_jv1gw0000gn/T/tmp958v_9nttfds
# I1108 14:57:40.645853 8280416384 dataset_info.py:669] Fields info.[release_notes, splits] from disk and from code do not match. Keeping the one from code.
# I1108 14:57:40.646153 8280416384 dataset_builder.py:593] Generating dataset glue (/Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0)
# Downloading and preparing dataset 298.29 MiB (download: 298.29 MiB, generated: 100.56 MiB, total: 398.85 MiB) to /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0...
# Dl Completed...: 0 url [00:00, ? url/s]          I1108 14:57:41.087229 8280416384 download_manager.py:400] Downloading https://dl.fbaipublicfiles.com/glue/data/MNLI.zip into /Users/briancruz/tensorflow_datasets/downloads/dl.fbaipublicfiles.com_glue_MNLIdNe8cK2kTBCG0bqBz2JxwShRT2KfuO3NVIwROTnjtfI.zip.tmp.86b93b636a6c4a5885063a7a027e203a...
# Extraction completed...: 100%|████████████████████████████| 12/12 [04:08<00:00, 20.74s/ file]
# Dl Size...: 100%|████████████████████████████████████████| 298/298 [04:08<00:00,  1.20 MiB/s]
# Dl Completed...: 100%|██████████████████████████████████████| 1/1 [04:08<00:00, 248.83s/ url]
# Generating splits...:   0%|                                       | 0/5 [00:00<?, ? splits/sI1108 15:03:32.576759 8280416384 writer.py:301] Done writing /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0.incomplete1B448E/glue-train.tfrecord*. Number of examples: 392702 (shards: [392702])
# Generating splits...:  20%|██████                        | 1/5 [01:42<06:50, 102.69s/ splitsI1108 15:03:35.216084 8280416384 writer.py:301] Done writing /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0.incomplete1B448E/glue-validation_matched.tfrecord*. Number of examples: 9815 (shards: [9815])
# Generating splits...:  40%|████████████▍                  | 2/5 [01:45<02:11, 43.82s/ splitsI1108 15:03:37.895513 8280416384 writer.py:301] Done writing /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0.incomplete1B448E/glue-validation_mismatched.tfrecord*. Number of examples: 9832 (shards: [9832])
# Generating splits...:  60%|██████████████████▌            | 3/5 [01:47<00:50, 25.04s/ splitsI1108 15:03:40.476293 8280416384 writer.py:301] Done writing /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0.incomplete1B448E/glue-test_matched.tfrecord*. Number of examples: 9796 (shards: [9796])
# Generating splits...:  80%|████████████████████████▊      | 4/5 [01:50<00:16, 16.17s/ splitsI1108 15:03:43.054614 8280416384 writer.py:301] Done writing /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0.incomplete1B448E/glue-test_mismatched.tfrecord*. Number of examples: 9847 (shards: [9847])
# Dataset glue downloaded and prepared to /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0. Subsequent calls will reuse this data.
# I1108 15:03:43.082206 8280416384 logging_logger.py:49] Constructing tf.data.Dataset glue for split train, from /Users/briancruz/tensorflow_datasets/glue/mnli/2.0.0
# /Users/briancruz/2023-fall-cs-194-294-merging-llms/.venv/lib/python3.8/site-packages/transformers/data/processors/glue.py:221: FutureWarning: This processor will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
#   warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)
# I1108 15:03:43.153598 8280416384 compute_fisher.py:47] Dataset loaded
# I1108 15:03:43.153666 8280416384 compute_fisher.py:49] Starting Fisher computation
# 2023-11-08 15:08:42.343013: W tensorflow/core/kernels/data/cache_dataset_ops.cc:854] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
# I1108 15:08:42.484767 8280416384 compute_fisher.py:52] Fisher computed. Saving to file...
# I1108 15:08:42.486102 8280416384 compute_fisher.py:55] Fisher saved to file
# All PyTorch model weights were used when initializing TFGPT2ForSequenceClassification.

# All the weights of TFGPT2ForSequenceClassification were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2ForSequenceClassification for predictions without further training.
# All PyTorch model weights were used when initializing TFGPT2ForSequenceClassification.

# All the weights of TFGPT2ForSequenceClassification were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFGPT2ForSequenceClassification for predictions without further training.
# I1108 15:08:50.986901 8280416384 dataset_info.py:578] Load dataset info from /Users/briancruz/tensorflow_datasets/glue/rte/2.0.0
# I1108 15:08:50.988744 8280416384 dataset_builder.py:528] Reusing dataset glue (/Users/briancruz/tensorflow_datasets/glue/rte/2.0.0)
# I1108 15:08:51.017008 8280416384 logging_logger.py:49] Constructing tf.data.Dataset glue for split validation, from /Users/briancruz/tensorflow_datasets/glue/rte/2.0.0
# /Users/briancruz/2023-fall-cs-194-294-merging-llms/.venv/lib/python3.8/site-packages/transformers/data/processors/glue.py:520: FutureWarning: This processor will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
#   warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)
# /Users/briancruz/2023-fall-cs-194-294-merging-llms/model_merging/model_merging/evaluation.py:7: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
#   return hfds.load_metric("glue", task)
# Merging coefficients: (1.0, 0.0)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.98, 0.020000000000000018)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.96, 0.040000000000000036)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.94, 0.06000000000000005)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.92, 0.07999999999999996)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.9, 0.09999999999999998)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.88, 0.12)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.86, 0.14)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.84, 0.16000000000000003)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.82, 0.18000000000000005)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.8, 0.19999999999999996)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.78, 0.21999999999999997)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.76, 0.24)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.74, 0.26)
# Scores:
#   accuracy: 0.296028880866426
# Merging coefficients: (0.72, 0.28)
# Scores:
#   accuracy: 0.296028880866426