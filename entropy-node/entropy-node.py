import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y,dtype=int)
    ans = 0.0;
    _, count = np.unique(y,return_counts = True)
    p = count/len(y)
    for i in range(len(p)):
        if(p[i]==0 ):
            continue 
        else:
            ans += p[i]*np.log2(p[i])
    return float(-ans)