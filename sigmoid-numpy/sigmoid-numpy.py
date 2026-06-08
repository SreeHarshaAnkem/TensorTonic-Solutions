import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    x = -1 * x
    exp_x = np.exp(x)
    denom = 1+exp_x
    return 1/denom