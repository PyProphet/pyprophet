from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = []
if use_cython:
    ext_modules += [
        Extension("pyprophet.scoring._optimized", ["pyprophet/scoring/_optimized.pyx"])
    ]
    ext_modules = cythonize(ext_modules)
else:
    ext_modules += [
        Extension("pyprophet.scoring._optimized", ["pyprophet/scoring/_optimized.c"])
    ]

setup(name="pyprophet", ext_modules=ext_modules, include_dirs=[numpy.get_include()])
