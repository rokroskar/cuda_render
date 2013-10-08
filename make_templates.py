


import numpy as np
from numpy import ceil, floor, sqrt, mod

def make_template(k) : 
    # total number of cells we need
    Ntotal = 1 + 4*k

    # total number of cells required for the template
    Ntemplate = ceil(sqrt(Ntotal))

    # need an odd number of cells
    if mod(Ntemplate,2) == 0 : 
        Ntemplate = Ntemplate + 1

    # number of cells in the base template -- if the number of total
    # cells equals the number of template cells, we're done

    if sqrt(Ntotal) == Ntemplate : 
        return np.ones((sqrt(Ntotal),sqrt(Ntotal)),dtype=np.float32)
    
    else : 
        template = np.zeros((Ntemplate,Ntemplate),dtype=np.float32)
        Nbase = Ntemplate-2
    
        # set the base to 1
        template[1:-1,1:-1] = 1
    
        Nleft = Ntotal - Nbase**2
        
#        # left-overs must be divisible by 4 and odd
        template[(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2,0] = 1
        template[(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2,-1] = 1
        template[0,(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2] = 1
        template[-1,(Ntemplate-Nleft/4)/2:-(Ntemplate-Nleft/4)/2] = 1
        
        return template


def calculate_distance(template, normalize=None) : 
    side_length = template.shape[0]
    # where is the center position
    cen = floor(side_length/2)
    
    for i in range(side_length) : 
        for j in range(side_length) : 
            template[i,j] *= sqrt((i-cen)**2 + (j-cen)**2)
    
    if normalize is not None: 
        template = template/template.max()*normalize
    
    return template
    
    
    
    

    
    
    
