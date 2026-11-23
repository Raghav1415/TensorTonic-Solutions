import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x = np.asarray(x, dtype = float)
    p = np.asarray(p, dtype = float)
    expectation = 0;

    for i in range(len(x)):
        expectation += x[i]*p[i]

    return expectation