# Austin's script to load in the Fisher information data and group it by layer.

import os
import sys
import numpy as np

cwd = os.getcwd()
cwd = '/Users/austinzane/PycharmProjects/2023-fall-cs-194-294-merging-llms/model_merging' # Comment out when running on command line
os.chdir(cwd)
sys.path.append(cwd)

from scripts.merge_and_evaluate import load_fishers
from model_merging import hdf5_util
from model_merging import hf_util
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


def load_models_v2(model_strs=['textattack/roberta-base-RTE']):
    models = []
    for i, model_str in model_strs:
        model_str = os.path.expanduser(model_str)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_str, from_pt=True
        )
        models.append(model)
        if i == 0:
            tokenizer = AutoTokenizer.from_pretrained(model_str)
    return models, tokenizer

def load_fishers_v2(fisher_dirs=['./fisher_coeffs_roberta_rte_mnli/rte_fisher.h5']):
    fishers = []
    for fisher_str in fisher_dirs:
        fisher_str = os.path.expanduser(fisher_str)
        fisher = hdf5_util.load_variables_from_hdf5(fisher_str, trainable=False)
        fishers.append(fisher)
    return fishers


def print_fishers(fisher_mat):
    n = len(fisher_mat) // 16
    # Initialize an array to store the results
    results = np.zeros((n, 3))  # 12 layers, 3 values per layer (average, sum, proportion)

    # Calculate the sum and average for each of the 12 layers
    for i in range(n):
        layer_matrices = fisher_mat[i * 16:(i + 1) * 16]  # Get the matrices for each layer
        layer_sum = sum(np.sum(matrix.numpy()) for matrix in layer_matrices)
        layer_avg = layer_sum / sum(matrix.numpy().size for matrix in layer_matrices)

        results[i, 0] = layer_avg
        results[i, 1] = layer_sum

    # Calculate the total sum
    total_sum = np.sum(results[:, 1])

    # Calculate the proportion of the total for each layer
    results[:, 2] = results[:, 1] / total_sum

    # Print the results
    print('Percent of total Fisher information for each layer:')
    for i in range(n):
        print(f'Layer {i + 1}: {results[i, 2] * 100:.2f}%')

    print('Total Fisher information for each layer:')
    for i in range(n):
        print(f'Layer {i + 1}: {results[i, 1]:.2f}')


# There are 197 weight matrices in the model. The first 192 are the weights of the 12 layers of the model.
# The next 5 are the weights of embeddings.
base_fisher, large_fisher = load_fishers_v2(['./fisher_coeffs_roberta_rte_mnli/mnli_fisher.h5', './fisher_coeffs_roberta_rte_mnli/large_rte_fisher.h5'])

# Get the trained model and tokenizer
rte_model, rte_tokenizer = load_models_v2()

# Only one model for now
rte_model = rte_model[0]

# The model has the main RoBERTa model and a task-specific classifier on top of it.
rte_model_roberta = rte_model.roberta

# The RoBERTa model has embeddings, an encoder, and a pooler. We are interested in the encoder for now.
rte_model_roberta_encoder = rte_model_roberta.encoder

# The encoder has 12 layers. Each layer has a self-attention layer and a feed-forward layer.
# There are 16 trainable weights per layer.
for i in range(16):
    print(rte_model_roberta_encoder.trainable_weights[i].name)

# Corresponding fisher information
for i in range(17):
    print(base_fisher[i].name)

for i in range(17):
    print(large_fisher[i].name)


print_fishers(base_fisher)
print_fishers(large_fisher)

# These are the embedding weights
for i in range(193, 197):
    print(rte_model_roberta.trainable_weights[i].name)


# Next, use roberta-large-mnli from huggingface.co/roberta-large-mnli
# or roberta-large-rte from huggingface.co/howey/roberta-large-rte

# Try to merge it with https://huggingface.co/roberta-large-mnli
