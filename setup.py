from setuptools import find_packages, setup

packages = find_packages(include=["numba_dpcomp"])

metadata = dict(
    name="numba-dpcomp",
    version="0.0.1",
    packages=packages,
    include_package_data=True,
)

setup(**metadata)
