from setuptools import setup, find_packages

setup(
    name="kadir",  # Kütüphanenizin adı
    version="0.1",  # Kütüphanenizin sürümü
    packages=find_packages(),  # Paketlerinizi bulur
    install_requires=[  # Hangi kütüphaneler gerekli
        "numpy",  
        "scipy",
        "matplotlib"
    ],
    author="M.Kadir",  # Kendi adınızı yazın
    author_email="22220030102@mersin.edu.tr",  # E-posta adresiniz
    description="GPU-Accelerated Numerical Methods Library",  # Kısa açıklama
    long_description=open('README.md').read(),  # Detaylı açıklama README'den gelir
    long_description_content_type='text/markdown',
    url="https://github.com/kadir/kadir",  # Kütüphanenizin GitHub URL'si (isteğe bağlı)
    classifiers=[  # Kütüphanenizin hangi kategorilere ait olduğunu belirler
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
)
