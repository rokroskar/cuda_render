import pynbody,sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm


def write_paramfile(stepnum):
    
    done = False
    info = open("output_%05d"%stepnum + "/info_%05d.txt"%stepnum)
    while(not done) : 
        line = info.readline().split()
    
        if len(line) > 0 : 
            if line[0] == 'aexp' : 
                aexp = float(line[2])
            elif line[0] == 'H0' : 
                H0 = float(line[2])
            elif line[0] == 'omega_l':
                omega_l = float(line[2])
            elif line[0] == 'omega_m' : 
                omega_m = float(line[2])
            elif line[0] == 'unit_l' : 
                unit_l = float(line[2])
            elif line[0] == 'unit_d' : 
                unit_d = float(line[2])
            elif line[0] == 'unit_t' : 
                unit_t = float(line[2])
                done = True

    pfile = open('run.param','w')
    g_to_msol = 5.0273993e-34
    cm_to_kpc = 3.2407793e-22
    cm_to_km = 1e-5
    s_to_gyr = 3.1688765e-17
    
    sysmass = unit_d*unit_l**3*g_to_msol # in Msol
    syslen = unit_l / aexp * cm_to_kpc # in kpc
    systime = np.sqrt(1.0/6.67e-8/(unit_d*aexp**3))*s_to_gyr # in Gyr
    sysvel = syslen/systime*0.97781311 # in km/s

    print "System mass = %e Msol"%sysmass
    print "System length = %e kpc"%syslen
    print "System time = %e"% systime

    pfile.write('dMsolUnit = %e\n'% sysmass)
    pfile.write('dKpcUnit = %e\n'% syslen)
    pfile.write('dHubble0 = %e\n'% (H0/100*syslen/sysvel/10.0))
    pfile.write('dLambda = %f\n'% omega_l)
    pfile.write('dOmega0 = %f\n'% omega_m)
    pfile.write('bComove = 1\n')
    pfile.close()

    
def make_gas_map(s, center=True):
    
    s.physical_units()
    if center:
        pynbody.analysis.angmom.sideon(s.g,mode='ssc',disk_size='1 kpc')
    plt.figure(figsize=(10,10))
    
    pynbody.plot.image(s.g,width=60,cmap=cm.Blues_r,units='Msol kpc^-2')
    plt.annotate('z = %.2f'%(1.0/s.properties['a']-1), (-25,25), color = "white",
                 weight='bold')
    plt.savefig(s.filename+'_gas_map.png', format='png')


if __name__ == '__main__':
    make_gas_map(load_ramses(int(sys.argv[1])))

