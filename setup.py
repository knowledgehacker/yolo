from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import imp

VERSION = imp.load_source('version', os.path.join('.', 'version.py'))
VERSION = VERSION.__version__

print(os.name)

ext_modules = [
    Extension("cython_utils.nms",
              sources=["cython_utils/nms.pyx"],
              libraries=["m"],  # Unix-like specific
              include_dirs=[numpy.get_include()]
              ),
    Extension("cython_utils.cy_yolo3_findboxes",
              sources=["cython_utils/cy_yolo3_findboxes.pyx"],
              libraries=["m"],  # Unix-like specific
              include_dirs=[numpy.get_include()]
              )
]

setup(
    version=VERSION,
    name='yolo',
    description='YOLO V3',
    license='GPLv3',
    url='https://github.com/knowledgehacker/yolo',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules)
)
