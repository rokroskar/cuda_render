




import numpy as np
import pylab as plt

def dotpx(x,y,z,a,b,rc,vc):
    return -vc**2*x/(x**2+y**2/a**2+z**2/b**2+rc**2)

def dotpy(x,y,z,a,b,rc,vc):
    return -vc**2/a**2*y/(x**2+y**2/a**2+z**2/b**2+rc**2)

def dotpz(x,y,z,a,b,rc,vc):
    return -vc**2/b**2*z/(x**2+y**2/a**2+z**2/b**2+rc**2)

def pot(x,y,z,a,b,rc,vc):
    return vc**2/2.0*np.log(x**2+y**2/a**2+z**2/b**2+rc**2)

def weave_integrate(R,p,a=1.,b=1.,rc=1.,vc=1.,ntime=100000,dt=.001) :
    from scipy import weave
    x,y,z, = R
    px, py, pz = p
   
    xs = np.empty(ntime,dtype='double')
    ys = np.empty(ntime,dtype='double')
    zs = np.empty(ntime,dtype='double')
    vxs = np.empty(ntime,dtype='double')
    vys = np.empty(ntime,dtype='double')
    vzs = np.empty(ntime,dtype='double')
    Es = np.empty(ntime,dtype='double')
    ts = np.linspace(0,ntime*dt,ntime)

    support_code = """
    #include <stdio.h>

    double dotpx(double x, double y, double z, double a, double b, double rc, double vc)
    {
        return -vc*vc*2.0*x/(x*x+y*y/a/a+z*z/b/b+rc*rc);
    }
    double dotpy(double x, double y, double z, double a, double b, double rc, double vc)
    {
        return -vc*vc*2.0*y/a/a/(x*x+y*y/a/a+z*z/b/b+rc*rc);
    }
    double dotpz(double x, double y, double z, double a, double b, double rc, double vc)
    {
        return -vc*vc*2.0*z/b/b/(x*x+y*y/a/a+z*z/b/b+rc*rc);
    }
    """

    code =\
    """
    double x0,y0,z0,px0,py0,pz0,a0,b0,rc0,vc0,dt0,x1,y1,z1,x2,y2,z2,px1,py1,pz1,px2,py2,pz2;
    int i;
    
    x0 = x;
    y0 = y;
    z0 = z;
    px0 = px;
    py0 = py;
    pz0 = pz;
    a0 = a;
    b0 = b;
    dt0 = dt;
    rc0 = rc;
    vc0 = vc;

    printf("%f %f %f %f %f %f %f %f %f\\n",x0,y0,z0,px0,py0,pz0,a0,dt0,rc0);

    for(i=0;i<ntime;i++) {
       x1 = x0 + px0*dt0;
       y1 = y0 + py0*dt0;
       z1 = z0 + pz0*dt0;
       
       px1 = px0 + dt0*dotpx(x0,y0,z0,a0,b0,rc0,vc0);
       py1 = py0 + dt0*dotpy(x0,y0,z0,a0,b0,rc0,vc0);
       pz1 = pz0 + dt0*dotpz(x0,y0,z0,a0,b0,rc0,vc0);
       
       x2 = x1 + px1*dt0;
       y2 = y1 + py1*dt0;
       z2 = z1 + pz1*dt0;
       px2 = px1 + dt0*dotpx(x1,y1,z1,a0,b0,rc0,vc0);
       py2 = py1 + dt0*dotpy(x1,y1,z1,a0,b0,rc0,vc0);
       pz2 = pz1 + dt0*dotpz(x1,y1,z1,a0,b0,rc0,vc0);
        
       x0 = .5*(x0+x2); 
       y0 = .5*(y0+y2);
       z0 = .5*(z0+z2);
       px0 = .5*(px0 + px2);
       py0 = .5*(py0 + py2);
       pz0 = .5*(pz0 + pz2);
        
       XS1(i) = x0;
       YS1(i) = y0;
       ZS1(i) = z0;
       VXS1(i) = px0;
       VYS1(i) = py0;
       VZS1(i) = pz0;
    }

    """

    vars = ['xs','ys','zs','vxs','vys','vzs', 'px','py','pz','x','y','z','dt','a','b','rc','vc','ntime']
    
    weave.inline(code,vars,support_code=support_code,compiler='gcc',force=1)

    return ts, xs, ys, zs, vxs, vys, vzs, Es

def integrate(R,p,a=1,b=1,rc=1.,vc=1.,ntime=100000,dt=.001) :
    x,y,z, = R
    px, py, pz = p

    xs = np.empty(ntime)
    ys = np.empty(ntime)
    zs = np.empty(ntime)
    vxs = np.empty(ntime)
    vys = np.empty(ntime)
    vzs = np.empty(ntime)
    Es = np.empty(ntime)
    ts = np.linspace(0,ntime*dt,ntime)
    
    for i,t in enumerate(ts):

        
        x1 = x + px*dt
        y1 = y + py*dt
        z1 = z + pz*dt
        px1 = px + dt*dotpx(x,y,z,a,b,rc,vc)
        py1 = py + dt*dotpy(x,y,z,a,b,rc,vc)
        pz1 = pz + dt*dotpz(x,y,z,a,b,rc,vc)
       
        x2 = x1 + px1*dt
        y2 = y1 + py1*dt
        z2 = z1 + pz1*dt
        px2 = px1 + dt*dotpx(x1,y1,z1,a,b,rc,vc)
        py2 = py1 + dt*dotpy(x1,y1,z1,a,b,rc,vc)
        pz2 = pz1 + dt*dotpz(x1,y1,z1,a,b,rc,vc)
        
        x = .5*(x+x2)
        y = .5*(y+y2)
        z = .5*(z+z2)
        px = .5*(px + px2)
        py = .5*(py + py2)
        pz = .5*(pz + pz2)
        
        xs[i] = x
        ys[i] = y
        zs[i] = z
        vxs[i] = px
        vys[i] = py
        vzs[i] = pz

        Es[i] = .5*(px**2+py**2 + pz**2) + pot(x,y,z,a,b,rc,vc)

    return ts, xs, ys, zs, vxs, vys, vzs, Es


def plot3d(R,p,a=1,b=1,rc=1.,vc=1.,ntime=100000,dt=.001) :
    import pylab as plt
    import mpl_toolkits.mplot3d.axes3d as p3

    fig=plt.figure()
    ax = p3.Axes3D(fig)
    
    t,x,y,z,vx,vy,vz,E = integrate(R,p,a,b,rc,vc,ntime,dt)
    
    ax.plot3D(x,y,z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim3d(-1,1)
    ax.set_aspect(1)
    plt.show()
    
    return ax

def plot(R,p,a=1.,b=1.,rc=1.,vc=1.,ntime=10000,dt=.001, ax = None, **kwargs) :
    

    t,x,y,z,vx,vy,vz,E = integrate(R,p,a,b,rc,vc,ntime,dt)

    if ax is None: 
        fig,ax=plt.subplots(1,2)
        ax[0].plot(x,y)
        ax[1].plot(np.sqrt(x**2+y**2),z)
    
    else : 
        ax.plot(x,y,**kwargs)
    
    

def find_crossing(t,y):
  
    inds = np.where(y[1:]*y[:-1] < 0)[0]

    zeros = []
    
    for i, ind in enumerate(inds): 
        
        sl = slice(ind-2,ind+2)
        if (np.diff(y[sl])>0).all() :
            zeros.append(np.interp(0.0,y[sl],t[sl]))
    return np.array(zeros)

def plot_sos(R,p,a=1,b=1,rc=1.,vc=1.,ntime=10000,dt=.01, ax = None, **kwargs):
    t,x,y,z,vx,vy,vz,E = integrate(R,p,a,b,rc,vc,ntime,dt)
    
    zeros = find_crossing(t,y)

    if ax == None: 
        fig, ax = plt.subplots(1,2)
        ax[0].plot(x,y)
        sosx = np.interp(zeros,t,x)
        sosy = np.interp(zeros,t,vx)
        ax[1].plot(sosx,sosy,'.')
        
        lims = [np.array([x,y]).min(),np.array([x,y]).max()]
        ax[0].set_xlim(lims)
        ax[0].set_ylim(lims)
        ax[0].set_aspect(1)

        lims = [np.array([sosx,sosy]).min(),np.array([sosx,sosy]).max()]
        ax[1].set_xlim(lims)
        ax[1].set_ylim(lims)
        ax[1].set_aspect(1)
        ax[0].set_xlabel('$x$')
        ax[0].set_ylabel('$y$')
        ax[1].set_ylabel('$p_x$')
        ax[1].set_xlabel('$x$')
    else : 
        ax.plot(np.interp(zeros,t,x),np.interp(zeros,t,vx),'.',**kwargs)


def create_sos(E,n=3,a=.9,b=1.,rc=.14,vc=1.,fmin=0.0,fmax=1.0,nf=5,ntime=100000) : 

    fracs = np.linspace(fmin,fmax,nf)
    
    f,axs = plt.subplots(5,figsize=(3,10))
    f2, axs2 = plt.subplots(1)

    vy0 = np.sqrt(2*(E-pot(3*rc,0,0,a,b,rc,vc)))
    KE0 = .5*vy0**2

    colors = ['b','g','r','c','y']
    for i,frac in enumerate(fracs) : 
        KEy = frac*KE0
        vy = np.sqrt(2*KEy)
        KEx = (1-frac)*KE0
        vx = np.sqrt(2*KEx)
       
        print vx,vy,KEx,KEy
        plot([3*rc,0,0],[vx,vy,0],a=a,rc=rc,ax=axs[i],ntime=ntime,color=colors[i])
        axs[i].set_xticklabels("")
        axs[i].set_yticklabels("")
        plot_sos([3*rc,0,0],[vx,vy,0],a=a,rc=rc,ax=axs2,ntime=ntime,color=colors[i])
        
    
    axs2.set_ylabel('$p_x$')
    axs2.set_xlabel('$x$')
