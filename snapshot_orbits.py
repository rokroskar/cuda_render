import pynbody
from pynbody import grav_omp
import numpy as np
import pickle
from galpy.potential import Potential
from scipy.interpolate import interp2d
import hashlib

class SnapshotPotential(Potential):
    """
    Create a snapshot potential object. The potential and forces are 
    calculated as needed through the _evaluate and _Rforce methods.
    
    **Input**:
    
    *s* : a simulation snapshot loaded with pynbody

    **Optional Keywords**:
    
    *num_threads* (4): number of threads to use for calculation

    """

    def __init__(self, s, num_threads=4) : 
        self.s = s
        self.point_hash = None
        self.pots = None
        self.rz_acc = None
        self._amp = 1.0
    
    def _evaluate(self, R,z,phi=None,t=None,dR=None,dphi=None) : 
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        new_point_hash = hashlib.md5(np.array([R,z])).hexdigest()

        if self.pots is None or new_point_hash != self.point_hash: 
            self.setup_potential(R,z)
            self.point_hash = hashlib.md5(np.array([R,z])).hexdigest()
            
        return self.pots
        
    def _Rforce(self, R,z,phi=None,t=None,dR=None,dphi=None) : 
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        new_point_hash = hashlib.md5(np.array([R,z])).hexdigest()

        if self.rz_acc is None or new_point_hash != self.point_hash: 
            self.setup_potential(R,z)
            self.point_hash = hashlib.md5(np.array([R,z])).hexdigest()

        return self.rz_acc[:,0]

        


    def setup_potential(self, R, z) : 
            
        # 
        # set up the four points per R,z pair to mimic axisymmetry
        # 
        points = np.zeros((len(R),len(z),4,3))
        
        for i in xrange(len(R)) :
            for j in xrange(len(z)) : 
                points[i,j] = [(R[i],0,z[j]),
                               (0,R[i],z[j]),
                               (-R[i],0,z[j]),
                               (0,-R[i],z[j])]

        points_new = points.reshape(points.size/3,3)
        pot, acc = grav_omp.direct(self.s,points_new,num_threads=4)

        pot = pot.reshape(len(R)*len(z),4)
        acc = acc.reshape(len(R)*len(z),4,3)

        # 
        # need to average the potentials
        #
        if len(pot) > 1:
            pot = pot.mean(axis=1)
        else : 
            pot = pot.mean()


        #
        # get the radial accelerations
        #
        rz_acc = np.zeros((len(R)*len(z),2))
        rvecs = [(1.0,0.0,0.0),
                 (0.0,1.0,0.0),
                 (-1.0,0.0,0.0),
                 (0.0,-1.0,0.0)]
        
        # reshape the acc to make sure we have a leading index even
        # if we are only evaluating a single point, i.e. we have
        # shape = (1,4,3) not (4,3)
        acc = acc.reshape((len(rz_acc),4,3))

        for i in xrange(len(R)) : 
            for j,rvec in enumerate(rvecs) : 
                rz_acc[i,0] += acc[i,j].dot(rvec)
                rz_acc[i,1] += acc[i,j,2]
        rz_acc /= 4.0
        
                
        self.pots = pot
        self.rz_acc = rz_acc


class SnapshotPotentialGrid(Potential): 
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


    def __init__(self, s,axisymmetric=True,
                 xlim=[-20,20],ylim=[-20,20],zlim=[-5,5],
                 nx=20,ny=20,nz=20,
                 num_threads=30) : 
        self.s = s 
        self.xlim = xlim 
        self.ylim = ylim 
        self.zlim = zlim
        self.nx = nx+1 
        self.ny = ny+1 
        self.nz = nz+1 
        self.num_threads = num_threads
        self._amp = 1.0
        self.axisymmetric = axisymmetric
        if axisymmetric : 
            self.pot_interpolant = None
            self.R_force_interpolant = None

        #self.generate_force_grid()


    def generate_force_grid(self, pkdgrav=False) : 
        try: 
            data = pickle.load(open(self.s._filename+'.potgrid'))
            self.pot = data['pot']
            self.acc = data['acc']
            self.grid = data['grid']
            self.rs = np.linspace(0,self.xlim[1],self.nx)
            self.zs = np.linspace(self.zlim[0],self.zlim[1],self.nz)

        except IOError: 
                    
            from pynbody.grav_omp import direct as direct_omp
            from os import system 

            s = self.s
            
            if self.axisymmetric: 
                
                s['pos_old'] = s['pos'].copy()
                phi = np.random.uniform(-1,1,len(s))
                s['x'] = s['rxy']*np.cos(phi)
                s['y'] = s['rxy']*np.sin(phi)
                
                rs = np.linspace(0,self.xlim[1],self.nx)
                zs = np.linspace(self.zlim[0],self.zlim[1],self.nz)
                
                self.gridshape = np.array((self.nx,self.nz,4,3))
            
                self.grid = np.empty(self.gridshape)
                self.xs = np.empty(self.nx*self.nz)
                self.ys = np.empty(self.nx*self.nz)

                for i, r in enumerate(rs) : 
                    for j,z in enumerate(zs) : 
                        self.grid[i,j] = [(r,0,z),(0,r,z),(-r,0,z),(0,-r,z)]
                                                
                s['pos'] = s['pos_old']
                self.rs = rs
                self.zs = zs

            else : 
                self.gridshape = np.array((self.ny,self.ny,self.nz,3))
                self.grid = np.empty(gridshape)
        
                xs = np.linspace(self.xlim[0],self.xlim[1],self.nx)
                ys = np.linspace(self.ylim[0],self.ylim[1],self.ny)
                zs = np.linspace(self.zlim[0],self.zlim[1],self.nz)
        
                for i,x in enumerate(xs) : 
                    for j,y in enumerate(ys) : 
                        for k,z in enumerate(zs) : 
                            self.grid[i,j,k] = np.array([x,y,z])
                self.xs = xs
                self.ys = ys
                self.zs = zs

            if not pkdgrav:
                    
                pot, acc = direct_omp(self.s,self.grid.reshape((self.gridshape[:-1].prod(),3)),
                                      num_threads=self.num_threads)
                
                if not self.axisymmetric: 
                    acc = acc.reshape((self.nx,self.ny,self.nz,3))
                    pot = pot.reshape((self.nx,self.ny,self.nz,1))
                    
                else : 
                    acc = acc.reshape((self.nx,self.nz,4,3))
                    pot = pot.reshape((self.nx,self.nz,4,1))
                    
            else : 
                sn = pynbody.snapshot._new(dm=len(self.s.d),gas=len(self.s.g),star=len(self.s.s)+self.nx*self.ny*self.nz)
                print self.nx*self.ny*self.nz
                sn['pos'][0:len(self.s)] = self.s['pos']
                sn['mass'][0:len(self.s)] = self.s['mass']
                sn['phi'] = 0.0
                sn['eps'] = 1e3
                sn['eps'][0:len(self.s)] = self.s['eps']
                sn['vel'][0:len(self.s)] = self.s['vel']
                sn['mass'][len(self.s):] = 1e-10
                sn['pos'][len(self.s):] = self.grid.reshape(self.nx*self.ny*self.nz,3)
                
                sn.write(fmt=pynbody.tipsy.TipsySnap, filename='potgridsnap')
                command = '~/bin/pkdgrav2_pthread -sz %d -n 0 +std -o potgridsnap -I potgridsnap +potout +overwrite'%self.num_threads
                print command
                system(command)
                sn = pynbody.load('potgridsnap')
                acc = sn['accg'][len(self.s):].reshape((self.nx,self.ny,self.nz,3))
                pot = sn['pot'][len(self.s):].reshape((self.nx,self.ny,self.nz))
                #            system('rm potgridsnap*')
            
            if self.axisymmetric : 
                acc_2d = np.zeros((self.nx,self.nz,2))
                pot_mean = np.zeros((self.nx,self.nz))
                for i in xrange(self.nx) : 
                    for j in xrange(self.nz) : 
                        pot_mean[i,j] = pot[i,j].mean()
                        for k,vec in enumerate([(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)]) :
                            vec = np.array(vec)
                            acc_2d[i,j,0] += np.dot(acc[i,j,k],vec)
                            acc_2d[i,j,1] += acc[i,j,k,2]
                acc = pynbody.array.SimArray(acc_2d,units=acc.units)
                pot = pynbody.array.SimArray(pot_mean, units=pot.units)
                
            self.pot = pot
            self.acc = acc
        
#            pickle.dump({'pot':self.pot,'acc':self.acc,'grid':self.grid},open(s._filename+'.potgrid','w',pickle.HIGHEST_PROTOCOL))

    def interpolate_forces(self,point,*args, **kwargs) :
        return trilinear_interpolate(point,self.grid,self.acc)

    def interpolate_potential(self,point,*args, **kwargs) :
        return trilinear_interpolate(point,self.grid,self.pot)

    def _evaluate(self, R, z, phi=None,t=None,dR=None,dphi=None) :
        if self.axisymmetric: 
            if self.pot_interpolant is None: 
                from scipy.interpolate import RectBivariateSpline
                self.pot_interpolant = RectBivariateSpline(self.rs,self.zs,self.pot)
            return self.pot_interpolant(R,z)[0]
        else : 
            if phi is None: 
                raise RuntimeError("Need to specify a phi for non-axisymmetric potential")
            x = R*np.cos(phi)
            y = R*np.sin(phi)
            return self.interpolate_potential((x,y,z))
    
    def _Rforce(self, R, z) : 
        if self.axisymmetric:
            rvec = np.array([R,z])
            rlen = np.sqrt(rvec.dot(rvec))
            rvec = rvec/rlen
        
            if self.R_force_interpolant is None : 
                from scipy.interpolate import RectBivariateSpline
                self.R_force_interpolant = RectBivariateSpline(self.rs,self.zs, self.acc[:,:,0])
                self.z_force_interpolant = RectBivariateSpline(self.rs,self.zs, self.acc[:,:,1])
        
            fvec = np.array([self.R_force_interpolant(R,z),self.z_force_interpolant(R,z)])

            return np.dot(fvec.reshape(2),rvec)
        else : 
            raise RuntimeError("non-axisymmetric potentials not supported")

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
        
    
