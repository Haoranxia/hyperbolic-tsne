""" Methods for initializing embedding.
"""
import numpy as np
from sklearn.decomposition import PCA

from sklearn.utils import check_random_state


def initialization(n_samples, n_components, X=None, method="random", random_state=None, init_scale=1e-4):
    """
    Generates an initial embedding.

    Parameters
    ----------
    n_samples : int
        Number of samples (points) of the embedding.
    n_components : int
        Number of components (dimensions) of the embedding.
    X : ndarray, optional
        High-dimensional points if using method=`pca`.
    method : string, optional
        Method to use for generating the initial embedding.
        Should be a string in [random, pca]
    random_state : int
        To ensure reproducibility (used in sklearn `check_random_state` function.
    init_scale: int
        Controls how "spread" out around the origin our random initial embedding will be.

    Returns
    -------
    X_embedded : ndarray
        array of shape (n_samples, n_components)
    """

    random_state = check_random_state(random_state)

    if method in ["pca"] and X is None:
        raise ValueError("The pca initialization requires the data X")

    if method == "random":
        X_embedded = init_scale * random_state.randn(n_samples, n_components).astype(np.float32)
    elif method == "pca":
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)
        X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        X_embedded /= np.std(X_embedded[:, 0]) * 10000  # Need to rescale to avoid convergence issues
    else:
        raise ValueError(f"Method of initialization `{method}` not supported. init' must be 'pca', 'random', or a numpy array")

    return X_embedded
