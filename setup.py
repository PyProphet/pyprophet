from setuptools import setup, find_packages
import sys, os

try:
    import multiprocessing
except:
    pass

version = '0.1'

setup(name='pyprophet',
    version=version,
    description="Python reimplementation of mProphet peak scoring",
    packages=find_packages(exclude=['ez_setup',
                        'examples', 'tests']),
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require="nose",
)
