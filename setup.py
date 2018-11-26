#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Installation script for onftools package '''
import re
import os
from os.path import join as pjoin, split as psplit, splitext

from setuptools import setup

import versioneer

# Get install requirements from requirements.txt file
with open('requirements.txt', 'rt') as fobj:
    install_requires = [line.strip() for line in fobj
                        if line.strip() and not line[0] in '#-']
# Get any extra test requirements
with open('test-requirements.txt', 'rt') as fobj:
    test_requires = [line.strip() for line in fobj
                     if line.strip() and not line[0] in '#-']

# Requires for distutils (only used in pypi interface?)
break_ver = re.compile(r'(\S+?)(\[\S+\])?([=<>!]+\S+)')
requires = [break_ver.sub(r'\1 (\3)', req) for req in install_requires]

cmdclass = versioneer.get_cmdclass()


setup(name='onftools',
      version=versioneer.get_version(),
      cmdclass=cmdclass,
      description='Utilities for processing exercises in notebooks',
      author='Matthew Brett',
      author_email='matthew.brett@gmail.com',
      maintainer='Matthew Brett',
      maintainer_email='matthew.brett@gmail.com',
      url='http://github.com/matthew-brett/onftools',
      packages=['onftools',
                'onftools.tests'],
      package_data = {'onftools': [
          'tests/data/*',
      ]},
      license='BSD license',
      classifiers = [
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
        ],
      long_description = open('README.rst', 'rt').read(),
      install_requires = install_requires,
      extras_require = {'test': test_requires},
      # For pip versions >= 9
      python_requires = '>=3.6',
      )
