from setuptools import setup, find_packages
from distutils.extension import Extension


min_numpy_version = (1, 9, 0)

try:
    import numpy
except ImportError:
    print "need at least numpy %d.%d.%d" % min_numpy_version
else:
    vtuple = tuple(map(int, numpy.__version__.split(".")))
    msg = "need at least numpy %s, found %s" % (min_numpy_version, vtuple)
    assert vtuple >= min_numpy_version, msg




######################################################################
version = (0, 21, 5)  # NEVER FORGET TO UPDATE version.py AS WELL !!!
######################################################################


ext_modules = [Extension("pyprophet._optimized", ["pyprophet/_optimized.c"])]

setup(name='pyprophet',
      version="%d.%d.%d" % version,
      author="Uwe Schmitt",
      author_email="rocksportrocker@gmail.com",
      description="Python reimplementation of mProphet peak scoring",
      license="BSD",
      url="http://github.com/uweschmitt/pyprophet",
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
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
          "numpy >= %d.%d.%d" % min_numpy_version,
          "pandas >= 0.17",
          "scipy >= 0.9.0",
          "numexpr >= 2.1",
          "scikit-learn >= 0.17",
          "matplotlib",
          "seaborn"
      ],
      entry_points={
          'console_scripts': [
              "pyprophet=pyprophet.main:main",
              ]
      },
      ext_modules=ext_modules,
      )
