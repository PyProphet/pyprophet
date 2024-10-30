import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

use_cython = True
ext = ".pyx" if use_cython else ".c"

extensions = [
    Extension(
        "pyprophet._optimized",
        [f"pyprophet/_optimized{ext}"],
        include_dirs=[numpy.get_include()],
    )
]

if use_cython:
    extensions = cythonize(extensions)

setup(
    ext_modules=extensions,
)

