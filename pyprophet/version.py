import pkg_resources  # part of setuptools
version = tuple(map(int, pkg_resources.require("pyprophet")[0].version.split(".")))
