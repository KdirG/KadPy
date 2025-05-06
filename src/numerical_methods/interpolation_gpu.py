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
    
    # Her bir x_new noktası için interpolasyon yap
    for i, x_point in enumerate(x_new_gpu):
        # Doğrusal interpolasyon için yakın noktaları bul
        mask = x_gpu <= x_point
        if not cp.any(mask):
            i0 = 0
        else:
            i0 = cp.where(mask)[0][-1]
        
        if i0 >= len(x_gpu) - 1:
            y_new[i] = y_gpu[-1]
            continue
        
        i1 = i0 + 1
        
        # Doğrusal interpolasyon formülü
        x0, x1 = x_gpu[i0], x_gpu[i1]
        y0, y1 = y_gpu[i0], y_gpu[i1]
        
        y_new[i] = y0 + (x_point - x0) * (y1 - y0) / (x1 - x0)
    
    # Sonucu geri döndür
    return y_new

def gpu_cubic_spline_interpolation(x, y, x_new):
    """
    GPU üzerinde çalışan kübik spline interpolasyon fonksiyonu
    """
    # Veriyi CPU'da hazırla
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    
    # Katsayıları al
    c0 = cs.c[0]
    c1 = cs.c[1]
    c2 = cs.c[2]
    c3 = cs.c[3]
    knots = cs.x
    
    # Katsayıları GPU'ya taşı
    c0_gpu = cp.asarray(c0)
    c1_gpu = cp.asarray(c1)
    c2_gpu = cp.asarray(c2)
    c3_gpu = cp.asarray(c3)
    knots_gpu = cp.asarray(knots)
    x_new_gpu = cp.asarray(x_new)
    
    # Sonuçları saklamak için dizi oluştur
    y_new = cp.zeros_like(x_new_gpu)
    
    # Her bir x_new noktası için spline değerini hesapla
    for i, x_point in enumerate(x_new_gpu):
        # x_point'in hangi aralıkta olduğunu bul
        mask = knots_gpu <= x_point
        if not cp.any(mask):
            idx = 0
        else:
            idx = cp.where(mask)[0][-1]
        
        if idx >= len(knots_gpu) - 1:
            idx = len(knots_gpu) - 2
        
        # Normalize edilmiş x değeri
        dx = x_point - knots_gpu[idx]
        
        # Kübik polinom değerini hesapla
        y_new[i] = c3_gpu[idx] + dx * (c2_gpu[idx] + dx * (c1_gpu[idx] + dx * c0_gpu[idx]))
    
    return y_new

# Daha verimli alternatif (linear interpolasyon için)
def gpu_linear_interpolation_vectorized(x, y, x_new):
    """
    GPU üzerinde çalışan daha verimli doğrusal interpolasyon fonksiyonu
    Vektörel işlemler kullanarak daha hızlı çalışır
    """
    # Girdi verilerini GPU'ya taşı
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)
    
    # Her x_new değeri için, x_gpu içindeki konumunu bul
    indices = cp.zeros(len(x_new_gpu), dtype=int)
    
    for i, x_val in enumerate(x_new_gpu):
        mask = x_gpu <= x_val
        if not cp.any(mask):
            indices[i] = 0
        else:
            indices[i] = cp.where(mask)[0][-1]
    
    # Sınır kontrolü
    valid_mask = indices < len(x_gpu) - 1
    
    # Geçerli indeks değerleri için interpolasyon hesapla
    i0 = indices[valid_mask]
    i1 = i0 + 1
    
    x0 = x_gpu[i0]
    x1 = x_gpu[i1]
    y0 = y_gpu[i0]
    y1 = y_gpu[i1]
    
    # Doğru x_new değerlerini seç
    x_points = x_new_gpu[valid_mask]
    
    # Sonuç dizisi oluştur
    y_new = cp.zeros_like(x_new_gpu)
    
    # Sondaki değerler için i0 = len(x_gpu) - 1 olan durumları ele al
    edge_mask = ~valid_mask
    if cp.any(edge_mask):
        y_new[edge_mask] = y_gpu[-1]
    
    # Interpolasyon hesapla
    y_new[valid_mask] = y0 + (x_points - x0) * (y1 - y0) / (x1 - x0)
    
    return y_new