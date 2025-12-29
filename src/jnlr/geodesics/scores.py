import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm


def generalized_energy_score(y_true: np.ndarray, y_samples: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Compute the generalized energy score (with alpha parameter).

    Parameters
    ----------
    y_true : array of shape (n, d)
        Observed true values.
    y_samples : array of shape (n, m, d)
        Forecast samples. m = number of samples.
    alpha : float, optional (default=1.0)
        Exponent in the generalized energy score (must be in (0, 2]).

    Returns
    -------
    energy_scores : array of shape (n,)
        The generalized energy score for each observation.
    """
    assert 0 < alpha <= 2, "alpha must be in (0, 2]"

    t, n, s = y_samples.shape
    y_true = y_true[:, np.newaxis, :]  # (t, 1, d)

    # First term: E[||y - Y||^alpha]
    term1 = np.linalg.norm(y_samples - y_true, axis=2) ** alpha
    term1 = term1.mean(axis=1)

    # Second term: E[||Y - Y'||^alpha]
    diffs = y_samples[:, :, np.newaxis, :] - y_samples[:, np.newaxis, :, :]  # (t, n, n, s)
    dists = np.linalg.norm(diffs, axis=3) ** alpha
    term2 = 0.5 * dists.mean(axis=(1, 2))
    return term1 - term2


def geodesic_score_p(geodesic_fun, y_true: jnp.ndarray, y_samples: jnp.ndarray):
    """
    Computes the sum of geodesic distances for probabilistic samples.

    Args:
        geodesic_fun: Function to compute geodesic distance, should return (distance, path).
        y_true: True values (n_samples, n_features).
        y_samples: Predicted samples (n_samples, n_pred_samples, n_features).
    Returns:
      Array of geodesic distances summed per observation.
    """
    distances = []
    for i in tqdm(range(y_true.shape[0])):
        point_distance = 0.
        for j in range(y_samples.shape[1]):
            d, _ = geodesic_fun(y_true[i], y_samples[i, j])
            point_distance += d
        distances.append(point_distance)

    return np.array(distances)