import diskfitting
import matplotlib.pylab as plt
from scipy import stats
import numpy as np

def sample_sech2(scale=1.,size=10000):
    out= np.zeros(size)
    nsamples= 0
        
    while nsamples < size:
        exps= stats.expon.rvs(scale=scale/2.,size=size-nsamples)
        comp= 1./np.cosh(exps/scale)**2./100.*np.exp(exps/scale*2.)
	indx= (stats.uniform.rvs(size=size-nsamples) < comp)
        nnewsamples= np.sum(indx)
        if nnewsamples > 0:
            newsamples= exps[indx]
            out[nsamples:nsamples+nnewsamples]= newsamples
            nsamples+= nnewsamples
    return out

def sample_double_sech2(scale1=1.0,scale2=2.0,f=0.5,size=10000) : 
    set1 = sample_sech2(scale=scale1,size=(1-f)*size)
    set2 = sample_sech2(scale=scale2,size=f*size)

    print len(set1), len(set2)

    sample = np.zeros(len(set1)+len(set2))
    sample[:len(set1)] = set1
    if f > 0 : 
        sample[len(set1):] = set2
    return sample

def sample_two_sech2(scale1=1.0,scale2=2.0,f=0.5,size=10000) : 
    out = np.zeros(size)
    nsamples = 0
    
    while nsamples < size: 
        exps= stats.expon.rvs(scale=scale2/2.,size=size-nsamples)
        #exps = stats.uniform.rvs(scale=5,size=size-nsamples)
        comp= two_sech2(exps,scale1=scale1,scale2=scale2,f=f)/(np.exp(-exps/scale2*2.)*100)
#        import pdb; pdb.set_trace()
        indx= (stats.uniform.rvs(size=size-nsamples) < comp)
        nnewsamples= np.sum(indx)
        if nnewsamples > 0:
            newsamples= exps[indx]
            out[nsamples:nsamples+nnewsamples]= newsamples
            nsamples+= nnewsamples
    return out

def sech2(xs,scale=1.0) : 
    return 1./np.cosh(xs/scale)**2

def two_sech2(xs,scale1=1.0,scale2=2.0,f=0.5) : 
    return (1.-f)*np.cosh(xs/scale1)**-2+f*np.cosh(xs/scale2)**-2

f=0.5
scale1=.5
scale2=1.5
sample = sample_two_sech2(scale1=scale1,scale2=scale2,f=f,size=10000)
h,bins = np.histogram(sample,bins=50)

bins = (bins[:-1]+bins[1:])/2.0

xs = np.linspace(0,5,100)

fit,num=diskfitting.two_comp_zfit_simple(sample,zmin=0,zmax=4)
fit2,num=diskfitting.two_comp_zfit_simple(sample,zmin=.1,zmax=4,func=diskfitting.negtwoexp)
res = diskfitting.mcerrors_simple_singlevar(sample,fit,0,4)
print fit
print fit2
print res
plt.figure()
plt.plot(xs,two_sech2(xs,scale1=scale1,scale2=scale2,f=f))
plt.plot(xs,two_sech2(xs,scale1=fit[0],scale2=fit[1],f=fit[2]),'--')
plt.plot(xs,(1-fit2[2])*np.exp(-xs/fit2[0])+fit2[2]*np.exp(-xs/fit2[1]),'--')
plt.plot(bins,h/float(h[0]))
plt.semilogy()
plt.show()

