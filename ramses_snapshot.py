from yt.mods import *

def get_sim_sphere(output) :

    sim = load(output+'/'+'info_'+output[-5:]+'.txt')

    v, c = sim.h.find_max("Density")

    print "ramses_snapshot: found max density at: %s" % c

    sp = sim.h.sphere(c, 3.0/sim["kpc"])
    
    com = sp.quantities["CenterOfMass"]()

    print "ramses_snapshot: center of mass = %s" % c

    L = sp.quantities["AngularMomentumVector"]()

    disk = sim.h.disk(c,L,10/sim['kpc'],1/sim['kpc'])

    return sim, sp, disk

if __name__ == '__main__':

    import getopt, sys, os, time

    try:
        opts, args = getopt.getopt(sys.argv[1:], "owf:a:p:m:", ["filepattern"])
    except:
        print 'bad options'
        #sys.exit(2)
        pass
    
    t1 = time.time()

    output = args[0]

    print "ramses_snapshot: loading %s" % output

    sim = load(output+'/'+'info_'+output[-5:]+'.txt')

    v, c = sim.h.find_max("Density")

    print "ramses_snapshot: found max density at: %s" % c

    sp = sim.h.sphere(c, 3.0/sim["kpc"])
    
    com = sp.quantities["CenterOfMass"]()

    print "ramses_snapshot: center of mass = %s" % c

    L = sp.quantities["AngularMomentumVector"]()

    print "ramses_snapshot: angular momentum vector = %s" % L

    # plot stuff

    pc = PlotCollection(sim,center=com)
    
    # add a slice plot using the angular momentum vector as the normal
    
    pc.add_cutting_plane("Density", L)

    pc.add_projection("Density", 0)
    pc.add_projection("Density", 1)
    pc.add_projection("Density", 2)

    pc.set_width(20, 'kpc')

    pc.save('pc1')

    t2 = time.time()

    print 'time elapsed: {0}'.format(t2-t1)


def make_plot_collection(sim) : 
    v, c = sim.h.find_max("Density")

    print "ramses_snapshot: found max density at: %s" % c

    sp = sim.h.sphere(c, 3.0/sim["kpc"])
    
    com = sp.quantities["CenterOfMass"]()

    print "ramses_snapshot: center of mass = %s" % c

    L = sp.quantities["AngularMomentumVector"]()

    print "ramses_snapshot: angular momentum vector = %s" % L

    # plot stuff

    pc = PlotCollection(sim,center=com)
    
    # add a slice plot using the angular momentum vector as the normal
    
    pc.add_cutting_plane("Density", L)

    pc.add_projection("Density", 0)
    pc.add_projection("Density", 1)
    pc.add_projection("Density", 2)

    pc.set_width(20, 'kpc')

    pc.save('pc1')
