# billiards: Program to model collisions between 2d discs
# Copyright (C) 2022  Tom Spencer (tspencerprog@gmail.com)
#
# This file is part of billiards
#
# billiards is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
