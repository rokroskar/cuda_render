import timeit


setup = """\
import pynbody, pynbody.grav_omp, direct
import numpy as np
s = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr_x0.5N/1/12M_hr_x0.5N.00100')

points = np.zeros((100,3))

points[:,0] = np.linspace(1,10,100)
"""
s1 = "res = pynbody.grav_omp.direct(s,points,eps=.1)"
t1 = timeit.Timer(s1,setup=setup)
print "pynbody omp: %.2f " % t1.timeit(number=5)

s2 = "res = pynbody.gravity.calc.direct(s,points,eps=.1)"
t2 = timeit.Timer(s2,setup=setup)
print "direct: %.2f " % t2.timeit(number=5)


s3 = "res = direct.direct_omp(s,points,eps=.1)"
t3 = timeit.Timer(s3,setup=setup)
print "pynbody omp: %.2f " % t3.timeit(number=5)

