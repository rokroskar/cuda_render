import numpy as np
import matplotlib.pylab as plt



def centroid(hdu, npix = 50):
    center = np.array((hdu[0].header.get('CRPIX1'), hdu[0].header.get('CRPIX2')))
    sx = slice(center[0]-npix/2,center[0]+npix/2)
    sy = slice(center[1]-npix/2,center[1]+npix/2)

    data = hdu[0].data

    xmax,ymax = np.where(data == data[sx,sy].max())

    sx = slice(xmax-npix/2,xmax+npix/2)
    sy = slice(ymax-npix/2,ymax+npix/2)

    xcenter, ycenter = 0.0, 0.0

    for i in np.arange(npix) - npix/2 + xmax:
        for j in np.arange(npix) - npix/2 + ymax:
            xcenter += data[i,j]*i
            ycenter += data[i,j]*j
    
    xcenter /= data[sx,sy].sum()
    ycenter /= data[sx,sy].sum()

    return xcenter, ycenter
