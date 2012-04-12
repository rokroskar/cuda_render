#!/usr/bin/python

import pynbody
import direct
import numpy as np


s = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr_x4N/1/12M_hr_x4N.00100')
    
points = np.zeros((2,3))
points[:,0] = np.linspace(1,100,2)
    
direct.direct(s,points,eps=.1)

