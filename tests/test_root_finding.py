# test_root_finding.py
import math
import cupy as cp
import numpy as np

def test_root_finding():
    # Test functions and their derivatives
    def f1(x):
        return x**3 - 2*x - 5
    
    def df1(x):
        return 3*x**2 - 2
    
    def f2(x):
        return xp.cos(x) - x
    
    def df2(x):
        return -xp.sin(x) - 1
    
    def f3(x):
        return xp.exp(x) - 2
    
    def df3(x):
        return xp.exp(x)
    
    # Test cases for both CPU and GPU
    for use_gpu in [False, True]:
        if use_gpu and not cp.is_available():
            print("CuPy not available, skipping GPU tests")
            continue
        
        backend = "GPU" if use_gpu else "CPU"
        print(f"\nTesting with {backend} backend")
        global xp
        xp = choose_backend(use_gpu)
        
        # Test bisection method
        print("\nTesting bisection method:")
        try:
            # Test 1
            root1 = bisection(f1, 1.0, 3.0, 1e-6, 100, use_gpu)
            print(f"Root of x³-2x-5: {root1} (expected ~2.094551)")
            
            # Test 2
            root2 = bisection(f2, 0.0, 1.0, 1e-6, 100, use_gpu)
            print(f"Root of cos(x)-x: {root2} (expected ~0.739085)")
        except Exception as e:
            print(f"Bisection test failed: {e}")
        
        # Test Newton-Raphson method
        print("\nTesting Newton-Raphson method:")
        try:
            # Test 1
            root1 = newton_raphson(f1, df1, 2.0, 1e-6, 100, use_gpu)
            print(f"Root of x³-2x-5: {root1} (expected ~2.094551)")
            
            # Test 2
            root2 = newton_raphson(f2, df2, 0.5, 1e-6, 100, use_gpu)
            print(f"Root of cos(x)-x: {root2} (expected ~0.739085)")
            
            # Test 3
            root3 = newton_raphson(f3, df3, 1.0, 1e-6, 100, use_gpu)
            print(f"Root of exp(x)-2: {root3} (expected ~0.693147)")
        except Exception as e:
            print(f"Newton-Raphson test failed: {e}")
        
        # Test error cases
        print("\nTesting error cases:")
        try:
            # Bisection with invalid interval
            bisection(f1, 0.0, 1.0, 1e-6, 100, use_gpu)
            print("FAIL: Should have raised ValueError for invalid interval")
        except ValueError as e:
            print(f"PASS: Correctly raised ValueError: {e}")
        
        try:
            # Newton-Raphson with zero derivative
            newton_raphson(lambda x: x**2, lambda x: 2*x, 0.0, 1e-6, 100, use_gpu)
            print("FAIL: Should have raised ValueError for zero derivative")
        except ValueError as e:
            print(f"PASS: Correctly raised ValueError: {e}")

if __name__ == "__main__":
    test_root_finding()