import numpy as np
import pynbody
import weakref
import math

class AbstractCamera(object) :
    def __init__(self, poi, camera, fov_degrees, up_vec=[0.0,1.0,0.0]) :
        self.poi = poi
        self.camera = camera
        self.fov_degrees = fov_degrees
        self.up_vec = up_vec
        
class Camera(object) :
    def __init__(self, f) :
        if hasattr(f,'camera') :
            self._original = f.camera._original
            self._last_xform = f.camera._last_xform
        else :
            self._original = {}
            self._last_xform = np.eye(4)
            for k, v in f.iteritems() :
                if len(v.shape)>1 :
                    print "Copying",k
                    self._original[k] = v.copy()

        self._kernel = pynbody.sph.Kernel2D()
        self.poi = [0,0,0]
        self.camera = [0,0,10.0]
        self.up_vec=[0,1.0,0]
        self.fov_degrees = 40.0
        self.sim = f
        self.nx = 1280
        self.ny = 720
        
        self.sim.camera = self

        
        

    def __repr__(self) :
        return "<Camera poi=%s camera=%s fov=%.1f deg (%.1f)>"%(str(self.poi),
                                                                str(self.camera),
                                                                self.fov_degrees,
                                                                self.fov)

    @staticmethod
    def _as3vec(x) :
        y = np.asarray(x)
        assert y.shape==(3,)
        return y

    @staticmethod
    def invert(M) :
        inv = np.eye(4)
        inv[:3,:3] = M[:3,:3].T
        inv[:3,3] = -np.dot(inv[:3,:3], M[:3,3])
        return inv
    
    @property
    def poi(self) :
        return self._poi

    @poi.setter
    def poi(self, x) :
        self._poi = self._as3vec(x)

    @property
    def camera(self) :
        return self._camera

    @camera.setter
    def camera(self, x) :
        self._camera = self._as3vec(x)


    @property
    def sim(self) :
        return self._sim()

    @sim.setter
    def sim(self, s) : 
        if s is not None :
            self._sim = weakref.ref(s)
        else :
            self._sim = lambda : None


    @property
    def fov(self) :
        return 2*self.z_cam*math.tan(self.fov_degrees*math.pi/360.0)
 
    @property
    def z_cam(self) :
        return math.sqrt(((self.poi-self.camera)**2).sum())

    @property
    def transform(self) :
        camera = -self.poi+self.camera
        R = pynbody.analysis.angmom.calc_faceon_matrix(camera, np.asarray(self.up_vec)/np.linalg.norm(self.up_vec))
        t = np.eye(4)
        t[:3,:3] = R
        t[:3,3] = np.dot(R,-self.poi)
        print R
        return t


    def __do_transform(self, dtrans, tol=1.e-8) :
        if np.max(np.abs(np.dot(dtrans,self.invert(self._last_xform))-np.eye(4)))>tol :
            for k, v in self._original.iteritems() :
                self.sim[k] = np.dot(dtrans[:3,:3],v.T).T
            self._last_xform = dtrans
            self.sim['pos']+=dtrans[:3,3]
        else :
            print "null-transform"
            
    def update_transform(self, acam=None) :
        if acam is not None :
            self.camera = acam.camera
            self.poi = acam.poi
            self.fov_degrees = acam.fov_degrees
            self.up_vec = acam.up_vec
            
        self.__do_transform(self.transform)
        

    def revert(self) :
        self.__do_transform(np.eye(4))
        
    
    @property
    def camera_dict(self) :
        camera_dict = {'z_camera': self.z_cam,
                       'x2': self.fov/2,
                       'kernel': self._kernel,
                       'nx': self.nx,
                       'ny': self.ny}
        return camera_dict
