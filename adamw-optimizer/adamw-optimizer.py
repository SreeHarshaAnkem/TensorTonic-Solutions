import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    m = beta1 * np.array(m) + (1-beta1) * np.array(grad)
    v = beta2 * np.array(v) + (1-beta2) * (np.array(grad) ** 2)

    # m_hat = m/(1-beta1**t)
    # v_hat = v/(1-beta2**t)
    
    update = m / (np.sqrt(v) + eps)
    w = np.array(w) - lr * weight_decay * np.array(w) - lr * update

    return w, m, v
    