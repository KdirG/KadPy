# integration.py

import numpy as np
from scipy.integrate import trapz,quad
from utils import choose_backend

 def trapezoidal_integral(x, y,use_gpu=None):
     """
    Compute the integral using the trapezoidal rule.
    
    Parameters:
        x (array): Array of x values (must be monotonically increasing)
        y (array): Array of y values corresponding to x
        use_gpu(bool):Whether to use GPU calculation
    Returns:
        float: Approximate integral value
    """
      xp = choose_backend(use_gpu)
       x_arr = xp.asarray(x)
    y_arr = xp.asarray(y)
    # Manual implementation of trapezoidal rule for GPU
        dx = x_arr[1:] - x_arr[:-1]
        y_sum = y_arr[:-1] + y_arr[1:]
        integral = xp.sum(dx * y_sum) / 2.0
        
        # Convert result to CPU (scalar value)
        if hasattr(integral, 'get'):
            return float(integral.get())
        return float(integral)
    else:
        # Use SciPy's implementation for CPU
        return trapz(y_arr, x_arr)

  def analytical_integral(func,a,b):
    """
    Compute the definite integral of a function using numerical quadrature.
    
    Parameters:
        func (callable): Function to integrate
        a (float): Lower bound of integration
        b (float): Upper bound of integration
        **kwargs: Additional arguments to pass to scipy.integrate.quad
        use_gpu (bool): Whether to use GPU acceleration
        num_points (int): Number of points for GPU integration
    Returns:
        tuple: (integral_value, absolute_error_estimate)
         Note: Error estimate is approximate for GPU calculations
    """
if use_gpu:
        try:
            import cupy as cp
            
            # Create a grid of points for integration
            x = cp.linspace(a, b, num_points)
            # Evaluate function at each point
            y = cp.asarray([func(xi) for xi in x.get()])
            
            # Use trapezoidal rule for integration
            integral = trapezoidal_integral(x, y, use_gpu=True)
            
            # Rough error estimate (proportional to step size squared)
            error_estimate = abs(integral) * ((b - a) / num_points)**2 / 12
            
            return integral, error_estimate
            
        except ImportError:
            print("CuPy not available. Using CPU integration instead.")
            use_gpu = False
    
    if not use_gpu:
        # Use SciPy's quad for CPU integration
        return quad(func, a, b)
        '''
        if __name__ == "__main__":
    # Define a test function
    def f(x):
        return np.sin(x)
    
    # Integration range
    a, b = 0, np.pi
    
    # Data points for trapezoidal integration
    x = np.linspace(a, b, 1000)
    y = np.sin(x)
    
    # CPU Integration
    trap_cpu = trapezoidal_integral(x, y, use_gpu=False)
    print(f"Trapezoidal (CPU): {trap_cpu}")
    
    anal_cpu, error_cpu = analytical_integral(f, a, b, use_gpu=False)
    print(f"Analytical (CPU): {anal_cpu} ± {error_cpu}")
    
    # GPU Integration (will fall back to CPU if CuPy isn't available)
    try:
        trap_gpu = trapezoidal_integral(x, y, use_gpu=True)
        print(f"Trapezoidal (GPU): {trap_gpu}")
        
        anal_gpu, error_gpu = analytical_integral(f, a, b, use_gpu=True)
        print(f"Analytical (GPU): {anal_gpu} ± {error_gpu}")
    except Exception as e:
        print(f"GPU integration failed: {e}")
        '''
