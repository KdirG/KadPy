# ode_solver.py
from scipy.integrate import odeint
def odeint_cpu(func, y0, t, args=(), atol=1e-6, rtol=1e-6, mxstep=500, h0=0.1, full_output=False, **kwargs):
    """
    Solve ODE using scipy.integrate.odeint on CPU.
    
    Parameters:
        func: callable
            The system of ODEs.
        y0: array-like
            Initial state.
        t: array-like
            Time points where solution is computed.
        args: tuple
            Additional arguments passed to func.
        atol: float
            Absolute tolerance for the solution.
        rtol: float
            Relative tolerance for the solution.
        mxstep: int
            Maximum number of steps to take.
        h0: float
            Initial step size.
        full_output: bool
            Whether to return additional output information.
        kwargs: additional keyword arguments passed to the solver.
        
    Returns:
        ndarray: Solution of the ODE at each time point.
    """
    return odeint(func, y0, t, args=args, atol=atol, rtol=rtol, mxstep=mxstep, h0=h0, full_output=full_output, **kwargs)