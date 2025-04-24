# README.md
#Numerical Methods Library with GPU Acceleration
This project is a Python library designed to provide a wide range of numerical methods, with CPU and GPU support. It includes methods for root finding, optimization, differentiation, integration, interpolation, and solving ordinary differential equations (ODEs). By leveraging both CPU (via NumPy) and GPU (via CuPy), the library aims to deliver optimized performance for large-scale computations.

Features
Root Finding: Methods like Newton-Raphson, Bisection.

Optimization: Implementations of Gradient Descent, Golden-Section, and others with CPU and GPU support.

Differentiation: Forward, Backward, and Central Difference methods for numerical differentiation.

Integration: Trapezoidal and Simpson's Rule for numerical integration.

Interpolation: Linear, Lagrange, and Cubic Spline interpolation.

ODE Solvers: Euler and Runge-Kutta methods for solving ordinary differential equations.

GPU Acceleration
The library utilizes CuPy to accelerate calculations on compatible GPUs. For operations on the CPU, it uses NumPy. Users can choose to run methods on the GPU, with automatic fallback to CPU if a GPU is unavailable.

Installation
To install the library, clone the repository and install the required dependencies:
git clone https://github.com/KdirG/GPUPy.git
cd GPUPy
