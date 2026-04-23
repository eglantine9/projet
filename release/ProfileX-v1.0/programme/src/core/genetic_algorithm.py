import numpy as np


def fuse_parents(matrix):
    """
    Geometrically average multiple latent vector rows into a single vector,
    enforcing Spherical interpolation (Norm Preservation).

    Parameters
    ----------
    matrix : numpy.ndarray
        Array containing all input vectors (shape: [num_variants, latent_dim])

    Returns
    -------
    new_image : numpy.ndarray
        A 1D numpy array representing the merged vector.

    Examples
    --------
    >>> fuse_parents(np.array([[0.5,0.8,0.2],[0.4,0.7,0.1]]))
    """
    # Mean across all vectors
    mean_vec = np.mean(matrix, axis=0)
    
    # Norm preservation to retain image sharpness and high-frequency features
    # Linear interpolation inside Gaussian priors pulls vectors to the origin (blur).
    norms = np.linalg.norm(matrix, axis=1)
    target_norm = np.mean(norms)
    current_norm = np.linalg.norm(mean_vec)
    
    if current_norm > 1e-6:
        return mean_vec * (target_norm / current_norm)
    return mean_vec


def mutate_vector(vector, num_variants, mutation_rate, mutation_level):
    """
    Generate multiple mutated variants of a given latent vector by substituting 
    certain dimensions with Gaussian noise.

    Parameters
    ----------
    vector : numpy.ndarray
        Base latent vector to mutate.
    num_variants : int
        Number of mutated alternatives to generate.
    mutation_rate : float
        Probability (0.0 to 1.0) of any given dimension mutating.
    mutation_level : float
        Standard deviation of the Gaussian noise added during mutation.

    Returns
    -------
    numpy.ndarray
        A 2D array containing `num_variants` rows of mutated vectors.
    """
    mutated_vectors = []
    for _ in range(num_variants):
        mutated_vec = vector.copy()
        mutation_probs = np.random.rand(len(vector)) 
        noise = np.random.normal(0, mutation_level, len(vector))
        for i in range(len(vector)):
            if mutation_probs[i] < mutation_rate:
                mutated_vec[i] += noise[i]
        mutated_vectors.append(mutated_vec)
    return np.array(mutated_vectors)
