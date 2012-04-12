from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("direct", 
                         ["direct_cython.pyx"],
                         libraries=["m"],
                         include_dirs=['/home/itp/roskar/python/numpy/core/include'],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'])
]

setup(
    name = 'direct gravity',
    cmdclass = {'build_ext': build_ext}, 
    ext_modules = ext_modules
)
