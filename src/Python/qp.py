"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import sympy as sp
import casadi as cs


def sympy2casadi(sympy_expr, sympy_var, casadi_var):
    # by jgillis
    # from https://gist.github.com/jgillis/80bb594a6c8fcf55891d1d88b12b68b8
    import casadi
    assert casadi_var.is_vector()
    if casadi_var.shape[1] > 1:
        casadi_var = casadi_var.T
    casadi_var = casadi.vertsplit(casadi_var)
    from sympy.utilities.lambdify import lambdify

    mapping = {'ImmutableDenseMatrix': casadi.blockcat,
               'MutableDenseMatrix': casadi.blockcat,
               'Abs': casadi.fabs
               }
    # f = lambdify(sympy_expr, sympy_var, modules=[mapping, casadi])
    f = lambdify(sympy_var, sympy_expr, modules=[mapping, casadi])
    print(casadi_var)
    return f(*casadi_var)


class Qp:

    def __init__(self, model, leg, **kwargs):

        self.L = np.array(model["linklengths"])
        self.leg = leg

    def qpcontrol(self, r_dd_des, x_in):
        M = self.leg.gen_M()
        C = self.leg.gen_C()
        G = self.leg.gen_G()
        B = np.zeros((4, 2))  # actuator selection matrix
        B[0, 0] = 1  # q0
        B[1, 2] = 1  # q2

        r_ddx = cs.SX.sym('r_ddx')  # ee acceleration
        r_ddz = cs.SX.sym('r_ddz')  # ee acceleration
        q0dd = cs.SX.sym('q0_dd')
        q1dd = cs.SX.sym('q1_dd')
        q2dd = cs.SX.sym('q2_dd')
        q3dd = cs.SX.sym('q3_dd')
        states = [r_ddx, r_ddz, q0_dd, q1_dd, q2_dd, q3_dd]  # state vector x
        n_states = len(states)  # number of states

        u0 = cs.SX.sym('u0')  # control torque 1
        u2 = cs.SX.sym('u2')  # control torque 2
        controls = [u0, u2]
        n_controls = len(controls)  # number of controls

        u = cs.SX.sym('u', n_controls)  # decision variables, control action matrix
        x = cs.SX.sym('x', n_states)  # represents the states over the opt problem.
        x_ref = cs.SX.sym('st_ref', n_states + n_states)  # initial and reference (desired) states

        q_dd = [q0_dd, q1_dd, q2_dd, q3_dd]

        sp.var('q0dd, q1dd, q2dd, q3dd')
        qdd_sp = sp.Matrix([q0dd, q1dd, q2dd, q3dd])
        M = sympy2casadi(qdd_sp, self.leg.gen_M(), q_dd)
        print("wut wut")
        constr_dyn = M + C + G - B*u + J.T * lam
        constr_D = D*qdd + d
        constr_dyn_fn = cs.Function('fn', [r_ddx, r_ddy, q0dd, q1dd, q2dd, q3dd, u0, u2], constr_dyn)
        constr_D_fn = cs.Function('fn', [r_ddx, r_ddy, q0dd, q1dd, q2dd, q3dd, u0, u2], constr_D)

        # --- calculate objective --- #
        r_dd = J + df
        obj = cs.mtimes((rdd - a_des).T, rdd - a_des)

        # --- calculate constraints --- #
        dyn_value = constr_dyn_fn(st[0], st[1], st[2], st[3], st[4], st[5],
                                  con[0], con[1])
        D_value = constr_dyn_fn(st[0], st[1], st[2], st[3], st[4], st[5],
                                con[0], con[1])
        constr = cs.vertcat(dyn_value, D_value)

        # --- set up solver --- #
        opt_variables = cs.vertcat(x, u)
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': x_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        # --- upper/lower bounds for constraint variables --- #
        c_length = np.shape(constr)[0]
        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # --- upper/lower bounds for optimization variables --- #
        o_length = np.shape(opt_variables)[0]
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints

        # --- setup is finished, now solve --- #
        u0 = np.zeros(n_controls)  # control inputs
        X0 = np.array(x_in)  # initialization of the state's decision variables
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables`
        x0 = cs.vertcat(X0, u0)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)
        u = np.array(sol['x'][n_states:]) # get controls from the solution

        return u
