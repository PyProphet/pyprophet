from setuptools import setup, find_packages
from distutils.extension import Extension


import numpy

version = (0, 13, 3)

ext_modules = [Extension("pyprophet._optimized", ["pyprophet/_optimized.c"])]

setup(name='pyprophet',
      version="%d.%d.%d" % version,
      author="Uwe Schmitt",
      author_email="rocksportrocker@gmail.com",
      description="Python reimplementation of mProphet peak scoring",
      license="BSD",
      url="http://github.com/uweschmitt/pyprophet",
      packages=find_packages(exclude=['ez_setup',
                                      'examples', 'tests']),
      include_package_data=True,
      include_dirs = [numpy.get_include()],
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
          "numpy >= 1.7.0",
          "pandas >= 0.13",
          "scipy >= 0.9.0",
          "numexpr >= 2.1",
          "scikit-learn >= 0.13",
      ],
      test_suite="nose.collector",
      tests_require="nose",
      entry_points={
          'console_scripts': [
              "pyprophet=pyprophet.main:main",
              ]
      },
      ext_modules=ext_modules,
      )
