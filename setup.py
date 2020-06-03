#!/usr/bin/python

import setuptools

setuptools.setup(
    name="ForeCite",
    version="0.0.1",
    url="https://github.com/allenai/ForeCite",
    packages=setuptools.find_packages(),
    install_requires=[],  # dependencies specified in requirements.in
    tests_require=[],
    zip_safe=False,
    test_suite="py.test",
    entry_points="",
)
