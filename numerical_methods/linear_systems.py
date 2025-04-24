#linear_systems.py

import numpy as np
from scipy.linalg import lu_factor,lu_solve

def solve_linear_system(A,b):
      """
    Solve a linear system Ax = b using numpy.linalg.solve

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        x: Solution vector
    """
    return np.linalg.solve(A, b)
    
    
def solve_linear_system_lu(A, b):
    """
    Solve a linear system Ax = b using LU decomposition.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        x: Solution vector
    """
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x    
    