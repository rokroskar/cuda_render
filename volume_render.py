from yt.mods import *

def volume_render(dd): 
   
    pf = dd.pf
    
    W = 200/pf['kpc'] # Width
    c = dd.center
    L = [1.0,1.0,1.0]
    field = 'nzDensity'
    lf = True
    N = 1024 

    mi, ma = na.log10(pf.h.all_data().quantities['Extrema']('Density')[0])

    dex = 4.0
    mi = ma - dex
# or 
#mi = -32.0
    def _newDens(field, data):
        new = data['Density']
        new[new<=0.0] = 10.**mi
        return new

    add_field("nzDensity", function=_newDens, take_log=True)

# Construct transfer function
    tf = ColorTransferFunction((mi-1, ma+1),nbins=1024)

# Sample transfer function with 5 gaussians.  Use new col_bounds keyword.
    Nc = 6

    tf.add_layers(Nc,w=0.01, col_bounds = (mi,ma), #alpha=na.logspace(-2, 0,Nc),
                  colormap='RdBu_r')

    cam = pf.h.camera(c, L, W, N, tf,
                      use_kd=True, sub_samples=5,
                      no_ghost=True,fields=[field],log_fields=[lf],
                      north_vector=[0.,0.,1.0], steady_north=True)

    image = cam.snapshot()

    return image, cam

