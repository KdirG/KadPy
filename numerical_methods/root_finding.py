# root_finding.py
from numerical_methods.utils import has_converged

def bisection(func, a, b, tolerance, max_iterations):
    """
    Manually implements the bisection method to find a root of a function.

    Args:
        func: The function for which to find the root (f(x)).
        a: The lower bound of the initial interval.
        b: The upper bound of the initial interval.
        tolerance: The desired accuracy (stopping criterion based on interval width).
        max_iterations: Maximum number of iterations to prevent infinite loops.

    Returns:
        The approximate root if found, or None if not found within max_iterations.
    """

    if func(a) * func(b) >= 0:
        raise ValueError("Function values at interval endpoints must have opposite signs.")

    iteration_count = 0
    while (b - a) / 2.0 > tolerance and iteration_count < max_iterations:
        c = (a + b) / 2.0  # Calculate midpoint
        if func(c) == 0:
            return c  # Exact root found (unlikely in practice, but good to check)
        elif func(a) * func(c) < 0:
            b = c      # Root is in [a, c]
        else:
            a = c      # Root is in [c, b]
        iteration_count += 1

    if iteration_count == max_iterations:
        print(f"Warning: Bisection method reached maximum iterations ({max_iterations}). Convergence may not be achieved within tolerance.")

    return (a + b) / 2.0  # Return midpoint as approximate root

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for finding a root of a function using its derivative.
    
    Args:
        f: Function whose root is to be found.
        df: Derivative of the function.
        x0: Initial guess.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        Approximate root value.

    Raises:
        ValueError: If derivative is zero or method fails to converge.
    """
    x = x0
    for _ in range(max_iter):
        dfx = df(x)
        if abs(dfx) < 1e-12:  # small derivate control
            raise ValueError("Derivative too close to zero; division by zero risk.")

        x_new = x - f(x) / dfx
        if has_converged(x, x_new, tol):
            return x_new
        x = x_new

    raise ValueError("Newton-Raphson did not converge.")
