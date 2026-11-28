from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    x = np.asarray(x,dtype=float)
    mean = np.mean(x,dtype=float)
    median = np.median(x)
    count = Counter(x)
    highest = max(count.values())
    mode = min(value for value, freq in count.items() if freq == highest)
    return {
        "mean" : mean,
        "median" : float(median),
        "mode" : float(mode)
    }
        
        