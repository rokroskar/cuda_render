cuda_render
===========

cuda-based renderer for simulation snapshots

Dependencies
------------

Uses [PyCuda](http://mathema.tician.de/software/pycuda/), [Pynbody](http://pynbody.github.io), and Numba, 
though the latter is not really necessary (just remove the `jit`/`autojit` statements).

First build the `radix_sort` extension, which wrapps the radix sort kernel from the [Nvidia CUB library](http://nvlabs.github.io/cub/):

```
> python setup_radix_sort.py build_ext --inplace 
```

To build the extension you need to have the cuda toolkit installed with `nvcc` in your `path`.

If the build and the compilation are successful, use the renderer: 

```
import pynbody
import cuda_render
import numpy as np
import numba
numba.codegen.debug.logger.setLevel(0)
import numba_template_render

s = pynbody.load(simulation)
image = cuda_render.cu_template_render_image(s.d,800,800,-.5,.5,timing=True, nthreads=128)
```

At the moment it's not particularly optimized yet... but further tweaks are coming soon!
