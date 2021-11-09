import os
import urllib.request

from setuptools import setup, find_packages

install_requires = [
    "pandas",
    "tensorflow == 2.3.0",
    "Image == 1.5.33",
    "torch",
    "torchvision",
    "humanfriendly == 9.2",
    "tqdm == 4.62.0",
    "matplotlib",
    "requests",
    "gdown",
    "jsonpickle"
]
setup(
    name="Owl_Classifier",
    version='1.0.0',
    packages=find_packages(where='src'),  
    install_requires=install_requires, 
    python_requires='>=3.5, <4',
)


