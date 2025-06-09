# from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize#,Extension
import numpy as np
setup(
    name = 'extensions',
    ext_modules = [Extension("extensions",
        ["extensions.pyx"],
    include_dirs=[np.get_include()])]
)

# setup(
#     name = 'extensions',
#     ext_modules = cythonize("extensions.pyx"),
#     include_dirs=[np.get_include()]
# )