import os
import urllib.request

from setuptools import setup, find_packages

install_requires = [
    "pandas == 1.1.5",
    "tensorflow == 2.3.0",
    "Image == 1.5.33",
    "torch == 1.9.0",
    "torchvision == 0.10.0",
    "humanfriendly == 9.2",
    "tqdm == 4.62.0",
    "jsonpickle == 2.0.0",
    "matplotlib == 3.3.4",
    "requests == 2.26.0",
    "gdown"
]
setup(
    name="Owl_Classifier",
    version='1.0.0',
    packages=find_packages(where='src'),  
    install_requires=install_requires, 
    python_requires='>=3.6, <4',
)


