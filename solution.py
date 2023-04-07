import pandas as pd
import numpy as np


chat_id = 297816265 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array) -> float:
    def log_likelihood(A, data):
        return np.sum(-A*26 + data*np.log(A*26) - np.log(np.math.factorial(data)))
    from scipy.optimize import minimize
    A_hat = minimize(lambda A: -log_likelihood(A, np.sum(x)), x0=1, method='BFGS').x[0] / 26
    
    mse = []
    for n in [10, 100, 1000]:
        sample = np.random.choice(x, size=n, replace=False)
        mse.append(np.mean((A_hat*n - sample)**2))
    
    thresholds = [0.00461, 0.00121, 0.000466]
    score = sum([1 if mse[i] < thresholds[i] else 0 for i in range(3)])

    return A_hat
