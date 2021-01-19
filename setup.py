import os
import sys

from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

install_requires = [
    'tensorflow~=1.15.3',
    'gym>=0.15.7',
    'plotly>=4.9.0',
    'optuna',
    'cloudpickle~=1.2.1',
    'numpy<1.19.0,>=1.16.0',
    'pandas~=1.1.3',
    'alive_progress',
    'PySimpleGUI',
]

setup(
    name='fault-tolerant-flight-control-drl',
    version='0.1.1',
    long_description=readme,
    url='https://github.com/kdally/fault-tolerant-flight-control-drl',
    license=license,
    author='kdally',
    author_email='k@dally.cc',
    description='Deep reinforcement learning for fault-tolerant flight control.',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    include_package_data=True,
)
