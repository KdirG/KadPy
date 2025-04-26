#linear_systems.py
import numpy as np
import cupy as cp
from utils import choose_backend
from scipy.linalg import lu_factor, lu_solve


def solve_linear_system(A, b, use_gpu=None):
    """
    Solve a linear system Ax = b
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        use_gpu: Boolean flag to indicate whether to use GPU (default: None)
    
    Returns:
        x: Solution vector
    """
    # Choose the appropriate backend
    xp = choose_backend(use_gpu)
    
    # Convert inputs to the selected backend's array format
    A_arr = xp.asarray(A)
    b_arr = xp.asarray(b)
    
    # Solve the system using the selected backend
    x_arr = xp.linalg.solve(A_arr, b_arr)
    
    # If using GPU, convert result back to CPU NumPy array
    if use_gpu:
        return cp.asnumpy(x_arr)
    else:
        return x_arr
    
def solve_linear_system_lu(A, b, use_gpu=None):
    """
    Solve a linear system Ax = b using LU decomposition.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        use_gpu: Boolean flag to indicate whether to use GPU (default: None)
        
    Returns:
        x: Solution vector
    """
    xp = choose_backend(use_gpu)
    
    # Convert inputs to the selected backend's array format
    A_arr = xp.asarray(A)
    b_arr = xp.asarray(b)
    
    if use_gpu:
        # Use CuPy's decomposition.lu
        P, L, U = cp.linalg.decomposition.lu(A_arr)
        
        # Solve Ly = Pb
        Pb = cp.dot(P, b_arr)
        y = cp.linalg.solve_triangular(L, Pb, lower=True)
        
        # Solve Ux = y
        x_arr = cp.linalg.solve_triangular(U, y, lower=False)
    else:
        # Use SciPy's LU factorization approach
        lu, piv = lu_factor(A_arr)
        x_arr = lu_solve((lu, piv), b_arr)
    
    # If using GPU, convert result back to CPU NumPy array
    if use_gpu:
        return cp.asnumpy(x_arr)
    else:
        return x_arr