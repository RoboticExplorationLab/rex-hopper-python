"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
# import sympy as sp
import casadi as cs
import itertools


class Qp:

    def __init__(self, **kwargs):
        pass
        # self.L = np.array(model["linklengths"])
        # self.leg = leg

    def qpcontrol(self, leg, r_dd_des, x_in, x_ref):
        # sp.var('q0dd, q1dd, q2dd, q3dd')
        # qdd_sp = sp.Matrix([q0dd, q1dd, q2dd, q3dd])
        M = leg.gen_M()
        C = leg.gen_C()
        G = leg.gen_G()
        J = leg.gen_jacEE()
        dee = leg.gen_dee()
        D = leg.gen_D()
        d = leg.gen_d()
        # cdot = leg.gen_cdot()  # only needed for sim baumgarte
        B = np.zeros((4, 2))  # actuator selection matrix
        B[0, 0] = 1  # q0
        B[2, 1] = 1  # q2

        # r_ddx = cs.SX.sym('r_ddx')  # ee acceleration
        # r_ddz = cs.SX.sym('r_ddz')  # ee acceleration
        q0dd = cs.SX.sym('q0dd')
        q1dd = cs.SX.sym('q1dd')
        q2dd = cs.SX.sym('q2dd')
        q3dd = cs.SX.sym('q3dd')
        lam1 = cs.SX.sym('lam1')
        lam2 = cs.SX.sym('lam2')
        states = [q0dd, q1dd, q2dd, q3dd, lam1, lam2]  # state vector x
        n_states = len(states)  # number of states

        u0 = cs.SX.sym('u0')  # control torque 1
        u2 = cs.SX.sym('u2')  # control torque 2
        controls = [u0, u2]
        n_controls = len(controls)  # number of controls

        u = cs.SX.sym('u', n_controls)  # decision variables, control action matrix
        x = cs.SX.sym('x', n_states)  # represents the states over the opt problem.
        st_ref = cs.SX.sym('st_ref', n_states + n_states)  # initial and reference (desired) states

        qdd = cs.vertcat(q0dd, q1dd, q2dd, q3dd)
        lam = cs.vertcat(lam1, lam2)
        eq1 = M @ qdd + C + G - B @ u - J.T @ lam
        eq2 = D @ qdd + d
        constr_dyn = [eq1[0], eq1[1], eq1[2], eq1[3]]  # conversion to dict
        constr_D = [eq2[0], eq2[1]]  # conversion to dict
        constr_dyn_fn = cs.Function('constr_dyn_fn', [q0dd, q1dd, q2dd, q3dd, lam1, lam2, u0, u2], constr_dyn)
        constr_D_fn = cs.Function('constr_D_fn', [q0dd, q1dd, q2dd, q3dd, lam1, lam2, u0, u2], constr_D)

        # --- calculate objective --- #
        r_dd = J @ qdd + dee
        obj = cs.mtimes((r_dd - r_dd_des).T, r_dd - r_dd_des)

        # --- calculate constraints --- #
        dyn_value = constr_dyn_fn(x[0], x[1], x[2], x[3], x[4], x[5], u[0], u[1])
        D_value = constr_D_fn(x[0], x[1], x[2], x[3], x[4], x[5], u[0], u[1])
        constr = cs.vertcat(dyn_value[0], dyn_value[1], dyn_value[2], dyn_value[3], D_value[0], D_value[1])

        # --- set up solver --- #
        opt_variables = cs.vertcat(x, u)
        qp = {'x': opt_variables, 'f': obj, 'g': constr}  #, 'p': st_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        # --- alt solver --- #
        # nlp = {'x': opt_variables, 'f': obj, 'g': constr}
        # solver = cs.nlpsol('solver', 'ipopt', nlp)

        # --- upper/lower bounds for constraint variables --- #
        c_length = np.shape(constr)[0]
        lbg = list(itertools.repeat(0, c_length))  # equality constraint
        ubg = list(itertools.repeat(0, c_length))  # equality constraint

        # --- upper/lower bounds for optimization variables --- #
        o_length = np.shape(opt_variables)[0]
        lbx = list(itertools.repeat(-100, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(100, o_length))  # input inequality constraints

        # --- setup is finished, now solve --- #
        u0 = np.zeros(n_controls)  # control inputs
        X0 = np.array(x_in)  # initialization of the state's decision variables
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(X0, u0)
        # sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)  # TODO: Is parameters needed?
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        u = np.array(sol['x'][n_states:])  # get controls from the solution

        return u
