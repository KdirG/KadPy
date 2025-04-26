# optimization.py
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from utils import choose_backend
def minimize_scalar_wrapper(func,use_gpu=None, method='brent', bounds=None, options=None):
     """
    Minimize a scalar function with optional GPU acceleration.
    
    Args:
        func: Function to minimize
        use_gpu: Whether to use GPU for function evaluation
        method: Optimization method ('brent', 'bounded', 'golden', etc.)
        bounds: Bounds for the search interval
        options: Additional options for the optimizer
    
    Returns:
        res: Optimization result
    """
    if use_gpu:
        # Create a wrapper function that moves data to GPU, evaluates, and returns to CPU
        def gpu_func(x):
            x_gpu = cp.asarray(x)
            result_gpu = func(x_gpu)  # Function should handle GPU arrays
            return float(cp.asnumpy(result_gpu))  # Convert back to scalar CPU value
        
        # Use SciPy's CPU optimizer but with GPU-accelerated function evaluations
        result = minimize_scalar(gpu_func, method=method, bounds=bounds, options=options)
    else:
        # Standard CPU optimization
        result = minimize_scalar(func, method=method, bounds=bounds, options=options)
    
    return result
    
 def minimize_wrapper(func, x0,use_gpu=None, method='BFGS', jac=None, bounds=None, constraints=None, options=None):
      """
    Minimize a scalar function with optional GPU acceleration.
    
    Args:
        func: Function to minimize
        use_gpu: Whether to use GPU for function evaluation
        method: Optimization method ('brent', 'bounded', 'golden', etc.)
        bounds: Bounds for the search interval
        options: Additional options for the optimizer
    
    Returns:
        res: Optimization result
    """
    if use_gpu:
        # Create a wrapper function that moves data to GPU, evaluates, and returns to CPU
        def gpu_func(x):
            x_gpu = cp.asarray(x)
            result_gpu = func(x_gpu)  # Function should handle GPU arrays
            return float(cp.asnumpy(result_gpu))  # Convert back to scalar CPU value
        
        # Use SciPy's CPU optimizer but with GPU-accelerated function evaluations
        result = minimize_scalar(gpu_func, method=method, bounds=bounds, options=options)
    else:
        # Standard CPU optimization
        result = minimize_scalar(func, method=method, bounds=bounds, options=options)
    
    return result