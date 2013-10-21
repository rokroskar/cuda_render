from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

sourcefiles = ['template_wrapper.pyx']
ext_modules = [Extension("template_wrapper", 
                          sourcefiles,
                         include_dirs = [numpy.get_include()]
                          )]

setup(
  name = 'Template render',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
