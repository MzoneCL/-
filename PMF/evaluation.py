import numpy as np

def get_rmse(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))