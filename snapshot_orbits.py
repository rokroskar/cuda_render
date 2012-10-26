
import pynbody
import numpy as np

class SnapshotPotential: 
    """
    Create a snapshot potential object. The potential and forces are
    calculated on a grid and stored. 

    **Input**:

    *s* : a simulation snapshot loaded with pynbody

    **Optional Keywords**:
    
    *xlim* ([-20,20]) : x limits
    
    *ylim* ([-20,20]) : y limits
    
    *zlim* ([-5,5])   : z limits
    
    *nx* (20): number of cells in x 
    
    *ny* (20): number of cells in y
 
    *nz* (20): number of cells in z

    *num_threads* (30): number of threads to use for calculation

    """


    def __init__(self, s,xlim=[-20,20],ylim=[-20,20],zlim=[-5,5],nx=20,ny=20,nz=20,num_threads=30) : 

        self.s = s 
        self.xlim = xlim 
        self.ylim = ylim 
        self.zlim = zlim
        self.nx = nx+1 
        self.ny = ny+1 
        self.nz = nz+1 
        self.num_threads = num_threads
        #self.generate_force_grid()


    def generate_force_grid(self, pkdgrav=False) : 
        from pynbody.grav_omp import direct as direct_omp
        from os import system 

        self.grid = np.empty((self.nx,self.ny,self.nz,3))
        
        xs = np.linspace(self.xlim[0],self.xlim[1],self.nx)
        ys = np.linspace(self.ylim[0],self.ylim[1],self.ny)
        zs = np.linspace(self.zlim[0],self.zlim[1],self.nz)
            
        for i,x in enumerate(xs) : 
            for j,y in enumerate(ys) : 
                for k,z in enumerate(zs) : 
                    self.grid[i,j,k] = np.array([x,y,z])
                    
        if not pkdgrav:
                    
            pot, acc = direct_omp(self.s,self.grid.reshape((self.nx*self.ny*self.nz,3)),
                                  num_threads=self.num_threads)
            acc = acc.reshape((self.nx,self.ny,self.nz,3))

            
        else : 
            sn = pynbody.snapshot._new(len(self.s)+self.nx*self.ny*self.nz)
            sn['pos'][0:len(self.s)] = self.s['pos']
            sn['mass'][0:len(self.s)] = self.s['mass']
            sn['eps'] = 0
            sn['eps'][0:len(self.s)] = self.s['eps']
            sn['mass'][len(self.s):] = 0.0
            sn['pos'][len(self.s):] = self.grid.reshape(self.nx*self.ny*self.nz,3)
            
            sn.write(fmt=pynbody.tipsy.TipsySnap, filename='potgridsnap')
            system('~/bin/pkdgrav.pthread -sz %d -dt 0 +std -o potgridsnap -I potgridsnap -binout 1'%self.num_threads)
            sn = pynbody.load('potgridsnap')
            acc = sn['accg'][len(self.s):].reshape(self.nx,self.ny,self.nz,3)
            pot = sn['pot'][len(self.s):].reshape(self.nx,self.ny,self.nz)
            system('rm potgridsnap*')
            
        self.xs, self.ys, self.zs = [xs,ys,zs]
        
        self.pot = pot
        self.acc = acc

    def interpolate_forces(self,point,*args, **kwargs) :
        return trilinear_interpolate(point,self.grid,self.acc)

    def interpolate_potential(self,point,*args, **kwargs) :
        return trilinear_interpolate(point,self.grid,self.pot)


def lininterp(x1,x2,fx1,fx2,x0) : 
    return fx2*(x0-x1)/(x2-x1)+fx1*(x2-x0)/(x2-x1)

def trilinear_interpolate(point,grid,arr):
    from scipy.interpolate import interp2d

    px, py, pz = point

    xs = grid[:,0,0].T[0]
    ys = grid[0,:,0].T[1]
    zs = grid[0,0,:].T[2]
    
    arr_int = np.zeros(arr.shape[-1])
    
    # find the relevant grid vertices

    xind = xs.searchsorted(px)
    yind = ys.searchsorted(py)
    zind = zs.searchsorted(pz)

    x1 = xs[xind-1]
    x2 = xs[xind]
    y1 = ys[yind-1]
    y2 = ys[yind]
    z1 = zs[zind-1]
    z2 = zs[zind]
        
    # do the bilinear interpolation on the top and bottom faces of the cube
        
    if len(arr_int) > 1 :
        for i in [0,1,2] : 
            arr_bot = [arr[xind-1,yind-1,zind-1,i],
                       arr[xind-1,yind,zind-1,i],
                       arr[xind,yind,zind-1,i],
                       arr[xind,yind-1,zind-1,i]]

            arr_top = [arr[xind-1,yind-1,zind,i],
                       arr[xind-1,yind,zind,i],
                       arr[xind,yind,zind,i],
                       arr[xind,yind-1,zind,i]]

            int_bot = interp2d([x1,x1,x2,x2],[y1,y2,y2,y1],arr_bot)
            int_top = interp2d([x1,x1,x2,x2],[y1,y2,y2,y1],arr_top)
        
            # interpolate in z

            arr_int[i] = lininterp(z1,z2,int_bot(px,py),int_top(px,py),pz)
         
    else : 
        arr_bot = [arr[xind-1,yind-1,zind-1],
                   arr[xind-1,yind,zind-1],
                   arr[xind,yind,zind-1],
                   arr[xind,yind-1,zind-1]]

        arr_top = [arr[xind-1,yind-1,zind],
                   arr[xind-1,yind,zind],
                   arr[xind,yind,zind],
                   arr[xind,yind-1,zind]]

        int_bot = interp2d([x1,x1,x2,x2],[y1,y2,y2,y1],arr_bot)
        int_top = interp2d([x1,x1,x2,x2],[y1,y2,y2,y1],arr_top)
        
        # interpolate in z

        arr_int = lininterp(z1,z2,int_bot(px,py),int_top(px,py),pz)
         

    return arr_int
            

#    def energy(self, q, p) : 
        
    
