# setup.py
from setuptools import setup, find_packages

setup(
    name='GPUPy',
    version='0.1.0',
    author='Kadir Göçer',
    author_email='22220030102@mersin.edu.tr',
    description='A numerical methods library with CPU and GPU support',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KdirG/GPUPy',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'scipy',
        'cupy-cuda12x',  # Eğer GPU opsiyonu sunuyorsan
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.13.3',
)
