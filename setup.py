# Compile using python setup.py build_ext --inplace
import glob
import os
from itertools import chain

import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from distutils.file_util import copy_file

setup(ext_modules=cythonize("billiards.pyx", compiler_directives={'language_level': 3}),
      include_dirs=[np.get_include(), "cpp/billiards"])

# Copy the binaries if they are newer
files_to_copy = chain(glob.glob("*.pyd"), glob.glob("*.so"))

for f in files_to_copy:
    copy_file(f, os.path.join("examples", f), update=1)
