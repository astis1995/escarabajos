import sys
import os

# Get the current directory
current_directory = os.getcwd()

# Add the current directory to the Python path
sys.path.append(current_directory)
print(current_directory)

from setuptools import setup, find_packages

setup(
    name='spectraltools',
    version='0.1',
    packages=find_packages(),
    # Optional: Include additional data files
    package_data={
        'mypackage': ['data/*.dat'],
    },
)