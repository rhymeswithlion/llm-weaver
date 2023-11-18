import numpy as np

def generate_weighted_vector(vec1, vec2, n):
    n_1 = len(vec1)
    n_2 = len(vec2)
    result = []

    def sample_from_vector(vector, center, spread=1.0):
        if len(vector) == 1:  # Handle the case where the vector has only one element
            return vector[0]
        probabilities = np.exp(-0.5 * ((np.arange(len(vector)) - center) / spread) ** 2)
        probabilities /= probabilities.sum()  # Normalize to make it a probability distribution
        return np.random.choice(vector, p=probabilities)

    for i in range(n):
        index1 = int((n_1-1) * (i // n))
        index2 = int((n_2-1) * (i // n))

        index1 = min(index1, n_1 - 1)
        index2 = min(index2, n_2 - 1)

        value1 = sample_from_vector(vec1, index1)
        value2 = sample_from_vector(vec2, index2)

        result.append(np.random.choice([value1, value2]))

    return np.array(result)


# Example usage
vec1 = np.array([1, 2, 3, 4, 5])
vec2 = np.array([10, 20, 30, 40])
output_length = 7

result_vector = generate_weighted_vector(vec1, vec2, output_length)
print("Result Vector:", result_vector)
