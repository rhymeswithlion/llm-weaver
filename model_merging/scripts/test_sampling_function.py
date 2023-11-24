import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import os
import sys
os.chdir('/Users/austinzane/PycharmProjects/2023-fall-cs-194-294-merging-llms/model_merging/scripts')
sys.path.append(os.getcwd())
from . import generate_weighted_vector


def plot_histograms(input_vector1, input_vector2, nruns, output_length=None, weight1=None, weight2=None,
                    alpha_blend=0.5, beta_choice=0.5, loc_spread=1.0):
    """
    Generates histograms based on the frequency of values in the middle index of
    the result vectors produced by 'generate_weighted_vector'.

    Parameters:
    - input_vector1, input_vector2, output_length, weight1, weight2, alpha_blend, beta_choice, loc_spread:
      Parameters for 'generate_weighted_vector'.
    - nruns: Number of times to run 'generate_weighted_vector'.
    """
    middle_indices1 = []
    middle_indices2 = []
    middle_index = (output_length if output_length is not None else len(input_vector1)) // 2

    for _ in range(nruns):
        result = generate_weighted_vector(input_vector1, input_vector2, output_length, weight1, weight2,
                                          alpha_blend, beta_choice, loc_spread)
        middle_indices1.append(result[middle_index] if result[middle_index] in input_vector1 else None)
        middle_indices2.append(result[middle_index] if result[middle_index] in input_vector2 else None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram for input_vector1
    count1 = Counter(middle_indices1)
    axes[0].bar(count1.keys(), count1.values())
    axes[0].set_title("Histogram for input_vector1")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    # Histogram for input_vector2
    count2 = Counter(middle_indices2)
    axes[1].bar(count2.keys(), count2.values())
    axes[1].set_title("Histogram for input_vector2")
    axes[1].set_xlabel("Value")

    # Set the same y-axis limits for both plots
    max_freq = max(max(count1.values(), default=0), max(count2.values(), default=0))
    axes[0].set_ylim(0, max_freq)
    axes[1].set_ylim(0, max_freq)

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_histograms(vec1, vec2, nruns=100, output_length=16, alpha_blend=1.0, beta_choice=0.5)
