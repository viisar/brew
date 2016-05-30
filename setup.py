#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://brew.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

import brew
VERSION=brew.__version__

setup(
    name='brew',
    version=VERSION,
    description='BREW: Python Multiple Classifier System API',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Dayvid Victor <victor.dvro@gmail.com>, Thyago Porpino <thyago.porpino@gmail.com>',
    author_email='brew-python-devs@googlegroups.com',
    url='https://github.com/viisar/brew',
    packages=find_packages(where='.', exclude=('test')),
    package_dir={'brew': 'brew'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='brew',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
    ],
)
