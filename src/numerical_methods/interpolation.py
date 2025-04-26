# interpolation.py
import numpy as np
from scipy.interpolate import CubicSpline,interp1d 
import matplotlib.pyplot as plt

def linear_interpolation(x,y,x_new):
    '''
    Making  interpolation for x values with using scipy interpolate
    
    Arguments:x(array)=>Given x values(indexes)
    y(array)=>Given y values(function values) 
    x_new(array)=>new x values which we are going to interpolate 
    
    Returns=>y_new=>  y values obtained with interpolation  
    '''
    #creating the interpolation function with scipy
     interp_func = scipy.interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")
    
    #interpolation for new x values
    y_new = interp_func(x_new)
    
    return y_new

    
 def spline_interpolation(x,y,x_new,bc_type):
     """
    Perform cubic spline interpolation using SciPy's CubicSpline.
    
    Args:
        x (array): Known x values (must be strictly increasing)
        y (array): Known y values
        x_new (array): New x values where interpolation is needed
        bc_type (str): Boundary condition type:
                      - 'natural': natural spline (second derivative = 0 at boundaries)
                      - 'clamped': first derivative specified at boundaries
                      - 'not-a-knot': continuous third derivative at first/last interior points
    
    Returns:
        array: Interpolated y values at x_new points
    """
     
    # Create cubic spline interpolator
     cs = CubicSpline(x, y, bc_type=bc_type)
     
     
     # Evaluate at new points
     y_new = cs(x_new)
     
     
     return y_new
     
     
    
