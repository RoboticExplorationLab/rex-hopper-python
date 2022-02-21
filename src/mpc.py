"""
Copyright (C) 2020-2022 Benjamin Bokser

Reference Material:

Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control
Donghyun Kim et al.

https://www.youtube.com/watch?v=RrnkPrcpyEA
https://github.com/MMehrez/ ...Sim_3_MPC_Robot_PS_obs_avoid_mul_sh.m
"""

import os
import numpy as np

import csv
import itertools

import casadi as cs

import utils

class Mpc:

    def __init__(self, **kwargs):

        self.u = np.zeros((4, 1))  # control signal
        self.dt = 0.025  # sampling time (s)
        self.N = 10  # prediction horizon
        # horizon length = self.dt*self.N = .25 seconds
        self.mass = float(12.12427)  # kg
        self.mu = 0.5  # coefficient of friction
        self.b = 40 * np.pi / 180  # maximum kinematic leg angle
        self.fn = None

        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        csv_body_path = 'res/spryped_urdf_rev06/spryped_data_body.csv'
        data_path_b = os.path.join(path_parent, csv_body_path)  # os.path.pardir
        with open(data_path_b, 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            next(data)  # skip headers
            values = list(zip(*(row for row in data)))  # transpose rows to columns
            values = np.array(values)  # convert list of nested lists to array

        ixx = values[1].astype(np.float)
        ixy = values[2].astype(np.float)
        ixz = values[3].astype(np.float)
        iyy = values[4].astype(np.float)
        iyz = values[5].astype(np.float)
        izz = values[6].astype(np.float)

        # m = np.zeros((6, 6))
        # m[0:3, 0:3] = np.eye(3) * self.mass
        i = np.zeros((3, 3))
        i[0, 0] = ixx
        i[0, 1] = ixy
        i[0, 2] = ixz
        i[1, 0] = ixy
        i[1, 1] = iyy
        i[1, 2] = iyz
        i[2, 0] = ixz
        i[2, 1] = iyz
        i[2, 2] = izz
        self.inertia = i  # inertia tensor in local frame

        self.r = np.array([0, 0, .1])  # vector from CoM to hip

    def mpcontrol(self, Q_base, x_in, x_ref, pf):
        dt = self.dt
        mass = self.mass

        R_base = utils.quat2rot(Q_base)  # get base rotation matrix
        phi = utils.quat2euler(Q_base)[0]
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        rz_phi = np.zeros((3, 3))
        rz_phi[0, 0] = c_phi  # Rz(phi)
        rz_phi[0, 1] = s_phi
        rz_phi[1, 0] = -s_phi
        rz_phi[1, 1] = c_phi
        rz_phi[2, 2] = 1

        # vector from CoM to hip in global frame (should just use body frame?)
        r_g = utils.Z(Q_base, self.r)
        # actual initial footstep position vector from CoM to end effector
        ri = pf + r_g

        # inertia matrix inverse
        # i_global = np.dot(np.dot(rz_phi, self.inertia), rz_phi.T)
        i_global = np.dot(np.dot(R_base, self.inertia), R_base.T)  # TODO: should this still be rz_phi?
        i_inv = np.linalg.inv(i_global)

        i11 = i_inv[0, 0]
        i12 = i_inv[0, 1]
        i13 = i_inv[0, 2]
        i21 = i_inv[1, 0]
        i22 = i_inv[1, 1]
        i23 = i_inv[1, 2]
        i31 = i_inv[2, 0]
        i32 = i_inv[2, 1]
        i33 = i_inv[2, 2]

        rz11 = rz_phi[0, 0]
        rz12 = rz_phi[0, 1]
        rz13 = rz_phi[0, 2]
        rz21 = rz_phi[1, 0]
        rz22 = rz_phi[1, 1]
        rz23 = rz_phi[1, 2]
        rz31 = rz_phi[2, 0]
        rz32 = rz_phi[2, 1]
        rz33 = rz_phi[2, 2]

        # r = foot position
        rx = ri[0]
        ry = ri[1]
        rz = ri[2]

        theta_x = cs.SX.sym('theta_x')
        theta_y = cs.SX.sym('theta_y')
        theta_z = cs.SX.sym('theta_z')
        p_x = cs.SX.sym('p_x')
        p_y = cs.SX.sym('p_y')
        p_z = cs.SX.sym('p_z')
        omega_x = cs.SX.sym('omega_x')
        omega_y = cs.SX.sym('omega_y')
        omega_z = cs.SX.sym('omega_z')
        pdot_x = cs.SX.sym('pdot_x')
        pdot_y = cs.SX.sym('pdot_y')
        pdot_z = cs.SX.sym('pdot_z')
        states = [theta_x, theta_y, theta_z,
                  p_x, p_y, p_z,
                  omega_x, omega_y, omega_z,
                  pdot_x, pdot_y, pdot_z]  # state vector x
        n_states = len(states)  # number of states

        f_x = cs.SX.sym('f1_x')  # controls
        f_y = cs.SX.sym('f1_y')  # controls
        f_z = cs.SX.sym('f1_z')  # controls
        controls = [f_x, f_y, f_z]
        n_controls = len(controls)  # number of controls

        gravity = -9.807

        # x_next = np.dot(A, states) + np.dot(B, controls) + g  # the discrete dynamics of the system
        x_next = [dt * omega_x * rz11 + dt * omega_y * rz12 + dt * omega_z * rz13 + theta_x,
                  dt * omega_x * rz21 + dt * omega_y * rz22 + dt * omega_z * rz23 + theta_y,
                  dt * omega_x * rz31 + dt * omega_y * rz32 + dt * omega_z * rz33 + theta_z,
                  dt * pdot_x + p_x,
                  dt * pdot_y + p_y,
                  dt * pdot_z + p_z,
                  dt * f_x * (i12 * rz - i13 * ry) + dt * f_y * (-i11 * rz + i13 * rx)
                  + dt * f_z * (i11 * ry - i12 * rx) + omega_x,
                  dt * f_x * (i22 * rz - i23 * ry) + dt * f_y * (-i21 * rz + i23 * rx)
                  + dt * f_z * (i21 * ry - i22 * rx) + omega_y,
                  dt * f_x * (i32 * rz - i33 * ry) + dt * f_y * (-i31 * rz + i33 * rx)
                  + dt * f_z * (i31 * ry - i32 * rx) + omega_z,
                  dt * f_x / mass + pdot_x,
                  dt * f_y / mass + pdot_y,
                  dt * f_z / mass + gravity + pdot_z]

        self.fn = cs.Function('fn', [theta_x, theta_y, theta_z,
                                     p_x, p_y, p_z,
                                     omega_x, omega_y, omega_z,
                                     pdot_x, pdot_y, pdot_z,
                                     f_x, f_y, f_z], x_next)  # nonlinear mapping of function f(x,u)

        u = cs.SX.sym('u', n_controls, self.N)  # decision variables, control action matrix
        st_ref = cs.SX.sym('st_ref', n_states + n_states)  # initial and reference states
        x = cs.SX.sym('x', n_states, (self.N + 1))  # represents the states over the opt problem.

        obj = 0  # objective function
        constr = []  # constraints vector

        k = 10
        Q = np.zeros((12, 12))  # state weighing matrix
        np.fill_diagonal(Q, [k, k, k, k, k, k, k, k, k, k, k, k])

        R = np.zeros((3, 3))  # control weighing matrix
        np.fill_diagonal(R, [k/2, k/2, k/2])

        constr = cs.vertcat(constr, x[:, 0] - st_ref[0:n_states])  # initial condition constraints
        # compute objective and constraints
        for k in range(0, self.N):
            st = x[:, k]  # state
            con = u[:, k]  # control action
            # calculate objective
            # why not just plug x_in and x_ref directly into st_ref??
            obj = obj + cs.mtimes(cs.mtimes((st - st_ref[n_states:(n_states * 2)]).T, Q),
                                  st - st_ref[n_states:(n_states * 2)]) + cs.mtimes(cs.mtimes(con.T, R), con)
            st_next = x[:, k + 1]
            f_value = self.fn(st[0], st[1], st[2], st[3], st[4], st[5],
                              st[6], st[7], st[8], st[9], st[10], st[11],
                              con[0], con[1], con[2])
            st_n_e = np.array(f_value)
            constr = cs.vertcat(constr, st_next - st_n_e)  # compute constraints

        # add additional constraints
        for k in range(0, self.N):
            constr = cs.vertcat(constr, u[0, k] - self.mu * u[2, k])  # f1x - mu*f1z
            constr = cs.vertcat(constr, -u[0, k] - self.mu * u[2, k])  # -f1x - mu*f1z

            constr = cs.vertcat(constr, u[1, k] - self.mu * u[2, k])  # f1y - mu*f1z
            constr = cs.vertcat(constr, -u[1, k] - self.mu * u[2, k])  # -f1y - mu*f1z

        opt_variables = cs.vertcat(cs.reshape(x, n_states * (self.N + 1), 1),
                                   cs.reshape(u, n_controls * self.N, 1))
        qp = {'x': opt_variables, 'f': obj, 'g': constr, 'p': st_ref}
        opts = {'print_time': 0, 'error_on_fail': 0, 'printLevel': "none", 'boundTolerance': 1e-6,
                'terminationTolerance': 1e-6}
        solver = cs.qpsol('S', 'qpoases', qp, opts)

        c_length = np.shape(constr)[0]
        o_length = np.shape(opt_variables)[0]

        lbg = list(itertools.repeat(-1e10, c_length))  # inequality constraints: big enough to act like infinity
        lbg[0:(self.N + 1)] = itertools.repeat(0, self.N + 1)  # IC + dynamics equality constraint
        ubg = list(itertools.repeat(0, c_length))  # inequality constraints

        # constraints for optimization variables
        lbx = list(itertools.repeat(-1e10, o_length))  # input inequality constraints
        ubx = list(itertools.repeat(1e10, o_length))  # input inequality constraints

        st_len = n_states * (self.N + 1)

        lbx[(st_len + 2)::3] = [0 for i in range(20)]  # lower bound on all f1z and f2z

        # setup is finished, now solve-------------------------------------------------------------------------------- #

        u0 = np.zeros((self.N, n_controls))  # 3 control inputs
        X0 = np.matlib.repmat(x_in, 1, self.N + 1).T  # initialization of the state's decision variables

        # parameters and xin must be changed every timestep
        parameters = cs.vertcat(x_in, x_ref)  # set values of parameters vector
        # init value of optimization variables
        x0 = cs.vertcat(np.reshape(X0.T, (n_states * (self.N + 1), 1)), np.reshape(u0.T, (n_controls * self.N, 1)))

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)

        solu = np.array(sol['x'][n_states * (self.N + 1):])
        # u = np.reshape(solu.T, (n_controls, self.N)).T  # get controls from the solution
        u = np.reshape(solu.T, (self.N, n_controls)).T  # get controls from the solution

        # u_cl = u[0, :]  # ignore rows other than new first row
        u_cl = u[:, 0]  # ignore rows other than new first row
        # ss_error = np.linalg.norm(x0 - x_ref)  # defaults to Euclidean norm
        # print("ss_error = ", ss_error)
        # print(u_cl)
        # print("Time elapsed for MPC: ", t1 - t0)

        return u_cl
