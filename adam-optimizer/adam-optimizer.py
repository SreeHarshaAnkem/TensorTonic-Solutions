import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m_t = beta1 * np.array(m) + (1-beta1) * np.array(grad)
    v_t = beta2 * np.array(v) + (1-beta2) * np.array(grad) ** 2

    m_hat = m_t / (1-beta1**t)
    v_hat = v_t / (1-beta2**t)

    param = param - lr * (m_hat/(np.sqrt(v_hat)  + eps))

    return param, m_t, v_t
 
    