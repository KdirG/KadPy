    # differentiation.py
    import numpy as np
    from gpu_support import gradient_gpu
    def compute_derivative(data,dx=1.0,method='auto',use_gpu=False)
     """
        Compute derivative of input data using different methods.
        
        Parameters:
            data: array-like
            dx: step size (default: 1.0)
            method: 'auto', 'forward', 'backward', 'central'
            use_gpu: True to use GPU-accelerated version if available

        Returns:
            Approximate derivative
        """

    if use_gpu: #GPU integration
    return gradient_gpu(data,dx)

    if method=='auto':
    return np.gradient(data,dx)
     elif method == 'forward':
            return forward_diff(data, dx)  # forward differentiation method
        elif method == 'backward':
            return backward_diff(data, dx)  # backward differentiation method
        elif method == 'central':
            return central_diff(data, dx)  # central differentiation method
        else:
            raise ValueError("Invalid method. Choose 'auto', 'forward', 'backward', or 'central'.")


    def forward_diff(f,x, h):
        """Forward difference method."""
        return (f(x + h) - f(x)) / h

    def backward_diff(f,x, h):
        """Backward difference method."""
       return (f(x) - f(x - h)) / h

    def central_diff(fx, h):
        """Central difference method."""
       return (f(x + h) - f(x - h)) / (2 * h)
        


    
    
    