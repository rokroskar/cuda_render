#!/bin/env python

from pytipsy import ifTipsy, ofTipsy, TipsyHeader, TipsyDarkParticle, TipsyGasParticle, TipsyStarParticle, tipsypos

class ParticleList:
    def __init__(self):
        self.values = []
        def append(self, value):
            self.values.append(value)
            def __getitem__(self, index):
                self.values.sort()
                return values[index]
            def __len__(self):
                return len(self.values)
            
