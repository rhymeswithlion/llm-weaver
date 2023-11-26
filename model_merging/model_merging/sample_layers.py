import numpy as np


def generate_fisher_distributions(fisher_matrices):
    """
    Generates a distribution of Fisher information over the layers of the model.
    NOTE: This function also calculates the total and average Fisher information for each layer.

    Parameters:
    - fisher_matrices: List of Fisher information matrices for each model.

    Returns:
    - list: List of numpy arrays containing the Fisher information distributions for each layer.
    """
    fisher_dists = []
    for fisher_mat in fisher_matrices:
        n = len(fisher_mat) // 16
        results = np.zeros((n, 3))

        for i in range(n):
            layer_matrices = fisher_mat[
                i * 16 : (i + 1) * 16
            ]  # Get the matrices for each layer
            layer_sum = sum(np.sum(matrix.numpy()) for matrix in layer_matrices)
            layer_avg = layer_sum / sum(
                matrix.numpy().size for matrix in layer_matrices
            )

            results[i, 0] = layer_avg
            results[i, 1] = layer_sum

        # Calculate the total sum
        total_sum = np.sum(results[:, 1])

        # Calculate the proportion of the total for each layer
        results[:, 2] = results[:, 1] / total_sum
        fisher_dists.append(results[:, 2])

    return fisher_dists


def sample_from_vector(vector, weight_vec, center, sampling_config, rng=None):
    """
    Selects a single element from a given vector based on a combination of location-based
    and provided weights.

    Parameters:
    - vector: List of integers to sample from.
    - weight_vec: Vector of weights corresponding to each element in 'vector'.
    - center: Center index for Gaussian-like location weighting.
    - alpha_blend: Blend factor between location-based weights and 'weight_vec'.
                    0 = pure 'weight_vec', 1 = pure location-based.
    - loc_spread: Spread of the Gaussian-like weighting (default is 1.0).

    Returns:
    - int: A randomly selected element from the input vector.
    """
    if len(vector) == 0:
        raise ValueError("Input vector is empty.")

    if len(vector) == 1:
        return vector[0]

    loc_probabilities = np.exp(
        -0.5 * ((np.arange(len(vector)) - center) / sampling_config["loc_spread"]) ** 2
    )
    loc_probabilities /= (
        loc_probabilities.sum()
    )  # Normalize to make it a probability distribution

    if rng is None:
        rng = np.random.default_rng(seed=42)
    return rng.choice(
        vector,
        p=sampling_config["alpha_blend"] * loc_probabilities
        + (1 - sampling_config["alpha_blend"]) * weight_vec,
    )


def generate_weighted_vector(
    input_vectors, weight_vectors=None, sampling_config=None, minus_one=True, rng=None
):
    """
    Generates a new vector of integers by selectively sampling from two input vectors.

    Parameters:
    - input_vector1, input_vector2: Input vectors of integers.
    - output_length: Desired length of the output vector. Defaults to the length of input_vector1.
    - weight1, weight2: Weight vectors for input_vector1 and input_vector2. Default to uniform weights.
    - alpha_blend: Blend factor for weighting schemes in sampling.
                    0 = pure weight_vec, 1 = pure location-based.
    - beta_choice: Blend factor for final choice between two vectors.
                    0 = pure input_vector1, 1 = pure input_vector2.

    Returns:
    - numpy.array: The resulting weighted vector.
    """
    if len(input_vectors) != 2:
        raise ValueError("Must provide exactly two input vectors.")

    input_vector1 = input_vectors[0]
    input_vector2 = input_vectors[1]

    if len(input_vector1) == 0 or len(input_vector2) == 0:
        raise ValueError("Input vectors must not be empty.")

    n_1, n_2 = len(input_vector1), len(input_vector2)
    # output_length = sampling_config[output_length if output_length is not None else n_1
    weight1 = weight_vectors[0] if weight_vectors is not None else np.ones(n_1) / n_1
    weight2 = weight_vectors[1] if weight_vectors is not None else np.ones(n_2) / n_2

    if rng is None:
        rng = np.random.default_rng(seed=42)

    result = []
    for i in range(sampling_config["output_length"]):
        if minus_one:
            index1 = int((n_1 - 1) * (i / (sampling_config["output_length"] - 1)))
            index2 = int((n_2 - 1) * (i / (sampling_config["output_length"] - 1)))
        else:
            index1 = int((float(n_1) * (i / sampling_config["output_length"])))
            index2 = int((float(n_2) * (i / sampling_config["output_length"])))

        value1 = sample_from_vector(
            input_vector1, weight1, index1, sampling_config, rng=rng
        )
        value2 = sample_from_vector(
            input_vector2, weight2, index2, sampling_config, rng=rng
        )

        result.append(
            rng.choice(
                a=[value1, value2],
                p=[sampling_config["beta_choice"], 1 - sampling_config["beta_choice"]],
            )
        )

    return np.array(result)


def generate_layer_config(fishers, sampling_config=None, **kwargs):
    """
    Generates a layer configuration for the new model.
    1. Load fisher matrices for each model.
    2. Generate a weight vector for each layer from fisher info.
    3. Generate a new layer config by sampling from the weight vectors.
    4. Return the new layer config.

    Returns:
    - dict: A dictionary containing the model configuration.
    """
    sampling_config = sampling_config if sampling_config is not None else {}
    sampling_config["alpha_blend"] = (
        sampling_config["alpha_blend"] if sampling_config is not None else 0.5
    )
    sampling_config["beta_choice"] = (
        sampling_config["beta_choice"] if sampling_config is not None else 0.5
    )
    sampling_config["loc_spread"] = (
        sampling_config["loc_spread"] if sampling_config is not None else 1.0
    )
    sampling_config["output_length"] = (
        sampling_config["output_length"] if sampling_config is not None else 12
    )

    fisher_dists = generate_fisher_distributions(fishers)

    layer_indices = []
    for i in range(len(fisher_dists)):
        layer_i = []
        n_i = len(fisher_dists[i])  # // 16
        # print(f"Model {i} has {n_i} layers.")
        for j in range(n_i):
            layer_i.append(f"model_{i}_layer_{j}")
        layer_indices.append(layer_i)

    layers = generate_weighted_vector(
        input_vectors=layer_indices,
        weight_vectors=fisher_dists,
        sampling_config=sampling_config,
        **kwargs,
    )

    model_config = {"layers": layers, "sampling_config": sampling_config}

    return model_config
