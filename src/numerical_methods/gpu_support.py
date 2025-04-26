# gpu_support.py
import cupy as cp



def gradient_gpu(data, dx):
    data = cp.asarray(data)
    return cp.gradient(data, dx)