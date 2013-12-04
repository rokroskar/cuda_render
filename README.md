cuda_render
===========

cuda-based renderer for simulation snapshots

Dependencies
------------

Uses (PyCuda)[http://mathema.tician.de/software/pycuda/], (Pynbody)[http://pynbody.github.io], and Numba, 
though the latter is not really necessary (just remove the jit/autojit statements).

To run:

'''
import pynbody
import cuda_render
import numpy as np
import numba
numba.codegen.debug.logger.setLevel(0)
import numba_template_render

s = pynbody.load(simulation)
image = cuda_render.cu_template_render_image(s.d,800,800,-.5,.5,timing=True, nthreads=128)

'''
