import sys
import numpy
from setuptools import setup, find_packages
from distutils.extension import Extension

ext_modules = [Extension("pyprophet._optimized", ["pyprophet/_optimized.c"])]

setup(name='pyprophet',
      version="2.0.2",
      author="Uwe Schmitt",
      author_email="rocksportrocker@gmail.com",
      description="PyProphet: Semi-supervised learning and scoring of OpenSWATH results.",
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
          "scipy >= 0.9.0",
          "pandas >= 0.17",
          "cython",
          "numexpr >= 2.1",
          "scikit-learn >= 0.17",
          "statsmodels >= 0.8.0",
          "matplotlib"
      ],
      entry_points={
          'console_scripts': [
              "pyprophet=pyprophet.main:cli",
              ]
      },
      ext_modules=ext_modules,
      )
