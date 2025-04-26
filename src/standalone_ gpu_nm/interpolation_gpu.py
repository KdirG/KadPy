import numpy as np 
import cupy as cp


def gpu_linear_interpolation(x, y, x_new):
    """
    GPU üzerinde çalışan doğrusal interpolasyon fonksiyonu
    """
    # Girdi verilerini GPU'ya taşı
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)
    
    # Sonuçları saklamak için dizi oluştur
    y_new = cp.zeros_like(x_new_gpu)
    
    # GPU üzerinde çalışacak kernel fonksiyonu
    @cp.fuse()
    def interpolate_point(x_point):
        # Doğrusal interpolasyon için yakın noktaları bul
        mask = x_gpu <= x_point
        if not cp.any(mask):
            i0 = 0
        else:
            i0 = cp.where(mask)[0][-1]
        
        if i0 >= len(x_gpu) - 1:
            return y_gpu[-1]
        
        i1 = i0 + 1
        
        # Doğrusal interpolasyon formülü
        x0, x1 = x_gpu[i0], x_gpu[i1]
        y0, y1 = y_gpu[i0], y_gpu[i1]
        
        return y0 + (x_point - x0) * (y1 - y0) / (x1 - x0)
    
    # Her yeni x değeri için interpolasyon uygula
    for i in range(len(x_new_gpu)):
        y_new[i] = interpolate_point(x_new_gpu[i])
    
    return y_new
    
 

def gpu_cubic_spline_interpolation(x, y, x_new):
    """
    Perform cubic spline interpolation on GPU using CuPy.
    
    Arguments:
        x (array): Known x values (must be strictly increasing)
        y (array): Known y values
        x_new (array): New x values where interpolation is needed
    
    Returns:
        y_new: Interpolated y values at x_new points
    """
    # Transfer data to GPU
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)
    
    # Number of data points
    n = len(x_gpu)
    
    # Step 1: Calculate the differences and coefficients
    h = cp.diff(x_gpu)
    
    # Calculate the right-hand side of the tridiagonal system
    alpha = cp.zeros(n-1)
    for i in range(1, n-1):
        alpha[i-1] = 3.0 * ((y_gpu[i+1] - y_gpu[i]) / h[i] - (y_gpu[i] - y_gpu[i-1]) / h[i-1])
    
    # Step 2: Solve the tridiagonal system for the second derivatives
    # Initialize arrays for tridiagonal system
    l = cp.ones(n)
    mu = cp.zeros(n-1)
    z = cp.zeros(n)
    
    # Forward sweep
    for i in range(1, n-1):
        l[i] = 2.0 * (x_gpu[i+1] - x_gpu[i-1]) - h[i-1] * mu[i-2] if i > 1 else 2.0 * (x_gpu[i+1] - x_gpu[i-1])
        mu[i-1] = h[i] / l[i]
        z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / l[i] if i > 1 else alpha[i-1] / l[i]
    
    # Back substitution
    c = cp.zeros(n)
    b = cp.zeros(n-1)
    d = cp.zeros(n-1)
    
    # Natural spline conditions: second derivatives at endpoints are zero
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1] if j < n-2 else z[j]
        b[j] = (y_gpu[j+1] - y_gpu[j]) / h[j] - h[j] * (c[j+1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j+1] - c[j]) / (3.0 * h[j])
    
    # Step 3: Evaluate the spline at each x_new point
    y_new_gpu = cp.zeros_like(x_new_gpu)
    
    # For each point in x_new, determine which segment it belongs to and evaluate
    for i in range(len(x_new_gpu)):
        # Find the right interval
        idx = cp.searchsorted(x_gpu, x_new_gpu[i]) - 1
        idx = cp.clip(idx, 0, n-2)  # Ensure index is in valid range
        
        # Compute the offset from the left boundary of the interval
        dx = x_new_gpu[i] - x_gpu[idx]
        
        # Evaluate the cubic spline polynomial
        y_new_gpu[i] = y_gpu[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
    
    return y_new_gpu

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Sample data
    x = np.linspace(0, 10, 10)
    y = np.sin(x)
    x_new = np.linspace(0, 10, 100)
    
    try:
        # GPU interpolation
        y_spline_gpu = gpu_cubic_spline_interpolation(x, y, x_new)
        
        # Convert back to CPU for plotting
        y_spline_cpu = cp.asnumpy(y_spline_gpu)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label='Data points')
        plt.plot(x_new, y_spline_cpu, '-r', label='GPU Cubic spline')
        plt.legend()
        plt.title('GPU-Based Cubic Spline Interpolation')
        plt.grid(True)
        plt.show()
        
        print("GPU interpolation successful!")
    except Exception as e:
        print(f"GPU interpolation failed: {e}")
        
        # Fall back to CPU implementation
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x, y)
        y_spline_cpu = cs(x_new)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label='Data points')
        plt.plot(x_new, y_spline_cpu, '-b', label='CPU Cubic spline (fallback)')
        plt.legend()
        plt.title('CPU-Based Cubic Spline Interpolation (Fallback)')
        plt.grid(True)
        plt.show()