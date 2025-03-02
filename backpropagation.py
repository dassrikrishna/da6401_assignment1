import numpy as np

def compute_grads(model, x, y):
    z, a = model.forward(x)
    y_cap = a[-1]
    error = y_cap - y        # del(L)/del(y_cap)
    grads_W, grads_b = [], []

    for k in reversed(range(len(model.weights))):
        delta = error * model.activation_derivative(z[k])   # del(L)/del(z)
        grads_W.insert(0, np.outer(a[k], delta))
        grads_b.insert(0, delta)
        
        if k > 0:
            error = np.dot(model.weights[k].T, delta)      # del(L)/del(a)
    
    return grads_W, grads_b
