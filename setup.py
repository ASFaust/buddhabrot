from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import os
import numpy

# Specify compilers
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

ext_modules = [
    Extension(
        "buddhabrot",
        ["buddhabrot.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]  # Include NumPy headers
    )
]

setup(
    name='Buddhabrot',
    ext_modules=cythonize(ext_modules),
)
