﻿import sys
import numpy as np
A=sys.path.insert(0, "..")
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

ext_module = Extension(
                        "extensions",
            ["extensions.pyx"],    #更改为自己想要转换的.pyx文件
            extra_compile_args=["/openmp"],
            extra_link_args=["/openmp"],
            )

setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [ext_module],
      include_dirs=[np.get_include()]
)