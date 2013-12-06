cuda_render
===========

cuda-based renderer for simulation snapshots

Dependencies
------------

Uses [PyCuda](http://mathema.tician.de/software/pycuda/), [Pynbody](http://pynbody.github.io), and Numba, 
though the latter is not really necessary (just remove the `jit`/`autojit` statements).

The build will first try to make the `radix_sort` extension, which wraps the GPU radix sort kernel from the [Nvidia CUB library](http://nvlabs.github.io/cub/). This also requires the cuda toolkit to be installed. Make sure that `nvcc` and the CUB directory are in your path. Install the package with the usual

```
> python setup.py install
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
