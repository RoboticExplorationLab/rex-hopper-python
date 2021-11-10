"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
# import sympy as sp
import cvxpy as cp
import itertools


class Cqp:

    def __init__(self, **kwargs):
        pass
        # self.L = np.array(model["linklengths"])
        # self.leg = leg

    def qpcontrol(self, leg, r_dd_des, x_in, x_ref):
        r_dd_des = r_dd_des.flatten()
        M = leg.gen_M()
        C = leg.gen_C().flatten()
        G = leg.gen_G().flatten()
        J = leg.gen_jacEE()
        dee = leg.gen_dee().flatten()
        D = leg.gen_D()
        d = leg.gen_d().flatten()
        B = np.zeros((4, 2))  # actuator selection matrix
        B[0, 0] = 1  # q0
        B[2, 1] = 1  # q2

        n_states = 6
        n_controls = 2
        x = cp.Variable(n_states)
        u = cp.Variable(n_controls)

        qdd = x[0:4]
        lam = x[4:]
        eq1 = M @ qdd + C + G - B @ u - J.T @ lam
        eq2 = D @ qdd + d

        # --- calculate objective --- #
        r_dd = J @ qdd + dee
        obj = 0.5*cp.sum_squares(r_dd - r_dd_des)

        # --- calculate constraints --- #
        constr = [0 == eq1]
        constr += [0 == eq2]
        # constr += [0 <= x]
        # constr += [100 >= x]
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve(solver=cp.SCS)  # , verbose=True)

        u = np.zeros(2) if u.value is None else u.value
        return u
