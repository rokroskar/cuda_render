#
#
# Generally useful stuff
#
#

import matplotlib.pylab as plt

def make_tile_figure(nrow,ncol,func,*args,**kwargs) : 

    fig = plt.figure(figsize=(5*ncol,5*nrow))

    for i in range(nrow) :
        for j in range(ncol) : 
            
            ax = fig.add_subplot(nrow,ncol,i*ncol+j+1)
            
            func(i*ncol+j,ax,*args,**kwargs)

            if (i < nrow-1) | (j > 0): clear_labels(ax)            
                
def clear_labels(ax):
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.set_xlabel('')
