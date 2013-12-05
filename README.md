cuda_render
===========

cuda-based renderer for simulation snapshots

Dependencies
------------

Uses [PyCuda](http://mathema.tician.de/software/pycuda/), [Pynbody](http://pynbody.github.io), and Numba, 
though the latter is not really necessary (just remove the `jit`/`autojit` statements).

First build the `radix_sort` extension, which wraps the GPU radix sort kernel from the [Nvidia CUB library](http://nvlabs.github.io/cub/) (this requires the cuda toolkit to be installed and `nvcc` in your path) :

```
> python setup_radix_sort.py build_ext --inplace 
```


If the build and the compilation are successful, use the renderer: 

```
import pynbody
import cuda_render
import numpy as np
import numba
numba.codegen.debug.logger.setLevel(0) # this is here because in the current version of the continuum
                                       # analytics distribution, the log levels are set to some 
                                       # annoyingly high value

s = pynbody.load(simulation)
image = cuda_render.cu_template_render_image(s.d,800,800,-.5,.5,timing=True, nthreads=128)
```

At the moment it's not particularly optimized yet... but further tweaks are coming soon!
