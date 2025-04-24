# optimization.py
import numpy as np
from scipy.optimize import minimize,minimize_scalar

def minimize_scalar_cpu(func, method='brent', bounds=None, options=None):
    """
    Minimize a scalar function using different methods (CPU).
    
    Arguments:
    func -- function to minimize
    method -- optimization method ('brent', 'bounded', 'golden', etc.)
    bounds -- bounds for the search interval (optional, for methods like 'bounded')
    options -- additional options for the optimizer
    
    Returns:
    res -- optimization result
    """
    result = minimize_scalar(func, method=method, bounds=bounds, options=options)
    return result
    
 def minimize_cpu(func, x0, method='BFGS', jac=None, bounds=None, constraints=None, options=None):
    """
    Minimize a function of several variables using different methods (CPU).
    
    Arguments:
    func -- function to minimize
    x0 -- initial guess
    method -- optimization method ('BFGS', 'Nelder-Mead', etc.)
    jac -- gradient of the function (optional)
    bounds -- bounds for variables (optional)
    constraints -- constraints on the variables (optional)
    options -- additional options for the optimizer
    
    Returns:
    res -- optimization result
    """
    result = minimize(func, x0, method=method, jac=jac, bounds=bounds, constraints=constraints, options=options)
    return result