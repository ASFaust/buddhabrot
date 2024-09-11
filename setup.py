from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("bhuddabrot.pyx"),  # Compiles the Cython file
    include_dirs=[np.get_include()]  # Ensures NumPy is available for compilation
)

