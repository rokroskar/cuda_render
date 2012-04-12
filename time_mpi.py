import mpi_obj
import pynbody
import datetime

def calc_pot(mec) : 
    start = datetime.datetime.now()
    mec.execute('import mpi_obj')
    mec.execute('import pynbody')
    start_load = datetime.datetime.now()
    mec.execute("s = mpi_obj.ParallelTipsySnap('erwingz400edt3.00512')")
    end_load = datetime.datetime.now()
    print 'file load took',end_load - start_load
    
    start_center = datetime.datetime.now()
    mec.execute("pynbody.analysis.halo.center(s)")
    end_center = datetime.datetime.now()
    print 'centering took', end_center - start_center
    
    start_sum = datetime.datetime.now()
    mec.execute("print s['mass'].sum()")
    end_sum = datetime.datetime.now()
    print 'sum over the whole simulation took ', end_sum - start_sum

    #start_profile = datetime.datetime.now()
    #mec.execute("p = pynbody.profile.Profile(s,min=0.001,max=15,nbins=50)")
    #end_profile = datetime.datetime.now()
    #print 'creating a profile took', end_profile - start_profile

    #start_pot = datetime.datetime.now()
    #mec.execute("print p['pot']")
    #end_pot = datetime.datetime.now()
    #print 'calculating potential took', end_pot - start_pot

    end = datetime.datetime.now()
    print 'full calculation took', end-start
