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
        M = leg.gen_M()
        C = leg.gen_C().flatten()
        G = leg.gen_G().flatten()
        # Jf = leg.gen_jacF()
        # df = leg.gen_df().flatten()
        Ja = leg.gen_jacA()
        da = leg.gen_da()
        D = leg.gen_D()
        dc = leg.gen_d().flatten()
        B = np.zeros((4, 2))  # actuator selection matrix
        B[0, 0] = 1  # q0
        B[2, 1] = 1  # q2

        n_states = 6
        n_controls = 2
        x = cp.Variable(n_states)
        u = cp.Variable(n_controls)
        qdd = x[0:4]
        lam = x[4:]
        eq1 = M @ qdd + C + G - B @ u - D.T @ lam
        eq2 = D @ qdd + dc

        # --- calculate objective --- #
        qdd_f = cp.vstack([x[0], x[2]])
        r_dd = Ja @ qdd_f + da
        P = np.eye(3)
        # obj = 0.5*cp.sum_squares(r_dd - r_dd_des)
        obj = 0.5 * cp.quad_form(r_dd - r_dd_des, P)

        # --- calculate constraints --- #
        constr = [0 == eq1]
        constr += [0 == eq2]
        constr += [-10 <= x]
        constr += [10 >= x]

        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(obj), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        u = np.zeros(2) if u.value is None else u.value

        # --- Check if result makes sense --- #
        qdd_new = np.linalg.solve(M, (B @ u - C - G))
        qdd_n = np.array([qdd_new[0], qdd_new[2]])
        Ja = leg.gen_jacA()
        da = leg.gen_da().flatten()
        print("rdd_new in task space = ", Ja @ qdd_n + da)

        return -u
