import numpy as np 
def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    den = np.power(10000, np.arange(0, d_model, 2) / d_model)

    res = np.zeros((seq_length, d_model))

    pos = np.arange(0, seq_length).reshape(-1,1)

    res[:, ::2] = np.sin(pos / den)
    res[:, 1::2] = np.cos(pos / den)

    return res
    

