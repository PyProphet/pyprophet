# encoding: utf-8
from __future__ import print_function

import pkg_resources  # part of setuptools
version = tuple(map(int, pkg_resources.require("pyprophet")[0].version.split(".")))
