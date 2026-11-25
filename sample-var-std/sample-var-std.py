import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x,dtype=float)
    mean = np.mean(x)
    variance = float(np.sum((x-mean)**2)/(len(x) - 1))
    deviation = float(np.sqrt(variance))
    answer = {
        "variance" : variance,
        "standard_deviation" : deviation 
    }
    return answer