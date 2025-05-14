# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except UnicodeDecodeError:
    long_description = "GPU Accelerated Numerical Methods Library"

setup(
    name="GPUPy",
    version="0.1.0",
    author="Adınız Soyadınız",
    author_email="email@example.com",
    description="GPU Accelerated Numerical Methods Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kullanici_adiniz/GPUPy",
    project_urls={
        "Bug Tracker": "https://github.com/kullanici_adiniz/GPUPy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=find_packages(),
    python_requires=">=3.13.3",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        'gpu': ['cupy>=10.0.0'],
    },
)