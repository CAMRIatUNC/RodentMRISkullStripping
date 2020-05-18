#!/usr/scripts/env python
"""
Rat Brain Masking with Keras 2D Unet
implemented by Li-Ming Hsu
"""
from distutils.core import setup
from setuptools import find_packages
import re, io

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('rbm/__init__.py', encoding='utf_8_sig').read()
    ).group(1)

__author__ = 'Not specified yet'
__email__ = ''
__maintainer__ = 'SungHo Lee'
__maintainer_email__ = 'shlee@unc.edu'
__url__ = ''

setup(name='rbm',
      version=__version__,
      description='Rat Brain Masking Tool with Keras 2D Unet',
      python_requires='>3.5, <3.8',
      author=__author__,
      author_email=__email__,
      maintainer=__maintainer__,
      maintainer_email=__maintainer_email__,
      url=__url__,
      license='GNLv3',
      packages=find_packages(),
      package_data={
          "": ["*.hdf5"],
      },
      install_requires=['keras>=2.3.0',
                        'SimpleITK>=1.2.4',
                        'numpy>=1.18.0',
                        'tensorflow>=2.0.0'
                        ],
      entry_points={
          'console_scripts': [
              'rbm=rbm.scripts.rbm:main',
          ],
      },
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Natural Language :: English',
            'Programming Language :: Python :: 3.7'
      ],
      keywords = 'rat fMRI brain masking'
     )
