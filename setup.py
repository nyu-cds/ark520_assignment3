from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'nbody_cython',
  ext_modules = cythonize("nbody_cython.pyx"),
)