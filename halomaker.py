import numpy as np
import pynbody
import matplotlib
import matplotlib.pyplot as plt


class HM_Halo:
    """

    This is a basic HaloMaker halo object. It is created from a treebrick file created by
    HaloMaker and contains the haloid, center, the velocity, the particle list

    """

    def __init__(self, haloid, pos, vel, plist):

        self.haloid = haloid
        self.pos = pos
        self.vel = vel
        self.plist = plist


class HM_
