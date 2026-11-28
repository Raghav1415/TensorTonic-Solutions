import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    x = np.asarray(x,dtype = float)
    y = np.asarray(y,dtype=float)
    distance = 0.0

    for i in range(len(y)):
        distance += np.abs(x[i] - y [i])

    return distance