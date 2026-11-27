import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    w = np.zeros(X.shape[1])
    b =0
    loss =0

    for i in range(steps):
        z = np.asarray((X@w)+b,dtype = float)
        prob = _sigmoid(z)
        for j in range(len(y)):
            loss += y[j]*np.log2(prob[j]) + (1-y[j])*np.log2(1-prob[j])

        loss = -(loss/len(y))
        grad_w = X.T @(prob-y)/len(y)
        grad_b = np.mean(prob-y)
        w = w - lr*grad_w
        b = b - lr*grad_b
    
    return w,float(b)