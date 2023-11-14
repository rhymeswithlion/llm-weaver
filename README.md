# CS194/294 Awesome Group Project

To see if your system is working

First, clean up any previous virtualenvs in .venv

```
$ make clean
=========================
 * Cleaning up
rm -rf .venv
```

Now create a new virtualenv and run the script `model_merging_isometric_fast.sh`

```
$ make .venv run-isometric-test

=========================
 * Creating conda environment
conda env create --prefix .venv/ -f environment.yml
Channels:
 - pytorch
 - conda-forge
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done


...


# To deactivate an active environment, use
#
#     $ conda deactivate

To activate with conda: conda activate /Users/briancruz/2023-fall-cs-194-294-merging-llms/.venv

=========================
 * Running isometric test
./scripts/model_merging_isometric_fast.sh
Using Python from /Users/briancruz/2023-fall-cs-194-294-merging-llms/scripts/../.venv/bin/python
model_merging package found: /Users/briancruz/2023-fall-cs-194-294-merging-llms/model_merging/model_merging/__init__.py
All PyTorch model weights were used when initializing TFRobertaForSequenceClassification.

All the weights of TFRobertaForSequenceClassification were initialized from the PyTorch model.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFRobertaForSequenceClassification for predictions without further training.
All PyTorch model weights were used when initializing TFRobertaForSequenceClassification.

...


2023-11-08 14:23:38.266385: W tensorflow/core/kernels/data/cache_dataset_ops.cc:854] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
Merging coefficients: (0.0, 1.0)
Scores:
  accuracy: 0.48
********************************************************************************
 Best Merge
********************************************************************************
Merging coefficients: (0.84, 0.16000000000000003)
Scores:
  accuracy: 0.72
```
