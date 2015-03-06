#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
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

setup(
    name='brew',
    version='0.1.0',
    description='BREW: Python Multiple Classifier System API',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Dayvid Victor <victor.dvro@gmail.com>, Thyago Porpino <thyago.porpino@gmail.com>',
    author_email='brew-python-devs@googlegroups.com',
    url='https://github.com/viisar/brew',
    packages=[
        'brew',
    ],
    package_dir={'brew': 'brew'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='brew',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
