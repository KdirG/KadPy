# numerical_methods/__init__.py

from .core import matrix_multiply
from .integration import trapezoidal_rule, simpsons_rule
from .differentiation import compute_derivative,forward_diff,backward_diff,central_diff
from .optimization import gradient_descent
from .root_finding import bisection,newton_raphson
from .interpolation import linear_interpolation,spline_interpolation
from .ode_solver import solve_ode
from .utils import timeit,relative_error, absolute_error,,has_converged,benchmark,to_gpu_array,to_cpu_array,compile_function_from_string
# __init__.py
