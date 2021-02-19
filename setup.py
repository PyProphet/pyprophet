import sys
import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension("pyprophet._optimized", ["pyprophet/_optimized.pyx"])]

# read the contents of README for PyPI
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pyprophet',
      version="2.1.7",
      author="The PyProphet Developers",
      author_email="rocksportrocker@gmail.com",
      description="PyProphet: Semi-supervised learning and scoring of OpenSWATH results.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      license="BSD",
      url="https://github.com/PyProphet/pyprophet",
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      include_dirs=[numpy.get_include()],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Chemistry',
      ],
      zip_safe=False,
      install_requires=[
          "Click",
          "numpy >= 1.9.0",
          "scipy",
          "pandas >= 0.17",
          "cython",
          "numexpr >= 2.1",
          "scikit-learn >= 0.17",
          "xgboost",
          "hyperopt",
          "statsmodels >= 0.8.0",
          "matplotlib",
          "tabulate"
      ],
      entry_points={
          'console_scripts': [
              "pyprophet=pyprophet.main:cli",
              ]
      },
      ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "2"}),
      )
