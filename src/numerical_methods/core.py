# core.py

import numpy as np
def matrix_multiply(A,B):
 A = np.array(A)
 B = np.array(B)
 if A.shape[1] != B.shape[0]:
    raise ValueError("Matris boyutları uyumsuz: A'nın sütun sayısı B'nin satır sayısına eşit olmalıdır.")

 return np.dot(A, B)
 