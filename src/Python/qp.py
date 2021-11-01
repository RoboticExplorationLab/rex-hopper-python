"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import sympy as sp
import csv
import os

#import transforms3d
#from sympy.physics.vector import dynamicsymbols

#from legbase import LegBase


class Leg(LegBase):

    def __init__(self, model, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            init_dq = [0., 0.]

        if init_q is None:
            init_q = [-30 * np.pi / 180, -150 * np.pi / 180]

        self.DOF = len(init_q)

        LegBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        self.L = np.array(model["linklengths"])