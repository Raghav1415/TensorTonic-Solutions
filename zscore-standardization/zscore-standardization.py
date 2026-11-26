import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.asarray(X,dtype=float)
    mean = np.mean(X,keepdims=True,axis=axis)
    std_deviation = np.std(X,keepdims=True,axis=axis)

    safe_std = np.where(std_deviation>eps, std_deviation,1.0)
    z_score = (X-mean)/safe_std
    return z_score

    