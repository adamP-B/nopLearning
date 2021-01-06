#! /usr/bin/python

import math

class Sos:
    "Second order statistics class"
    def __init__(self):
        self.zero()
    def zero(self):
        self.aver = 0.0
        self.nvar = 0.0
        self.count = 0.0
    def av(self):
        return self.aver
    def number(self):
        return int(self.count)
    def add(self, x):
        delta = (x-self.aver)/(self.count+1.0)
        self.nvar = self.nvar + self.count*delta*(x-self.aver)
        self.aver = self.aver + delta
        self.count = self.count + 1
    def var(self):
        if self.count>1.0:
            return self.nvar/(self.count-1.0)
        else:
            return 0.0
    def sd(self):
        if self.var>0.0:
            return math.sqrt(self.nvar)
        else:
            return 0.0
    def err(self):
        if self.var>0.0:
            return math.sqrt(self.nvar/self.count)
        else:
            return 0.0

