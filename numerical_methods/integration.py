# integration.py

import numpy as np
from scipy.integrate import trapz,quad

 def trapezoidal_integral(x, y):
     """
    Compute the integral using the trapezoidal rule.
    
    Parameters:
        x (array): Array of x values (must be monotonically increasing)
        y (array): Array of y values corresponding to x
        
    Returns:
        float: Approximate integral value
    """
     
     
return trapz(y,x)


  def analytical_integral(func,a,b):
    """
    Compute the definite integral of a function using numerical quadrature.
    
    Parameters:
        func (callable): Function to integrate
        a (float): Lower bound of integration
        b (float): Upper bound of integration
        **kwargs: Additional arguments to pass to scipy.integrate.quad
        
    Returns:
        tuple: (integral_value, absolute_error_estimate)
    """
integral,error=quad(func,a,b)
 return integral,error  
