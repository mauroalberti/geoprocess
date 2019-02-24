#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='geoprocess',
    version='0.0.1',
    packages=find_packages(),

    install_requires=['numpy', 'mplstereonet', 'matplotlib', 'pygsf'],

    # metadata for upload to PyPI
    author="Mauro Alberti",
    author_email="alberti.m65@gmail.com",
    description="geoprocess, a library for processing of field-map geological data",
    license="GPL-3",
    keywords="structural geology",
    url="https://github.com/mauroalberti/geoprocess",
    project_urls={
        "Bug Tracker": "https://github.com/mauroalberti/geoprocess/issues",
        "Documentation": "https://github.com/mauroalberti/geoprocess/blob/master/README.md",
        "Source Code": "https://github.com/mauroalberti/geoprocess/tree/master/process",
    }
)
