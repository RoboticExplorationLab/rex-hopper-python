"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
import utils


class Mpc:

    def __init__(self, model, t, N, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.m = m  # kg
        self.Ad = np.eye(12)
        self.Ad[3:6, 9:12] = np.eye(3)*t
        self.Bd = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [t / m, 0, 0],
                            [0, t / m, 0],
                            [0, 0, t / m]])
        self.N = N  # prediction horizon
        self.mu = mu  # coefficient of friction
        self.g = g
        self.G = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -g]).T
        self.n_x = np.shape(self.Ad)[1]
        self.n_u = np.shape(self.Bd)[1]
        self.Q = np.eye(self.n_x)
        self.R = np.eye(self.n_u)
        np.fill_diagonal(self.Q, [1., 1., 1., .1, .1, .1, 1., 1., 1., .1, .1, .1])  # TODO: change
        np.fill_diagonal(self.R, [0., 0., 0.])
        self.rh = np.array(model["r_h"])  # vector from CoM to hip
        '''
        i_val = model["body_inertia"]
        ixx = i_val[0]
        ixy = i_val[1]
        ixz = i_val[2]
        iyy = i_val[3]
        iyz = i_val[4]
        izz = i_val[5]
        '''
        [ixx, ixy, ixz, iyy, iyz, izz] = model["body_inertia"]
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
        self.I_b = i  # inertia tensor in body frame

    def mpcontrol(self, X_in, X_ref, Q_base, C):
        t = self.t
        N = self.N
        m = self.m
        mu = self.mu
        n_x = self.n_x
        n_u = self.n_u
        Ad = self.Ad
        Bd = self.Bd
        Rz = utils.rz_phi(Q_base)
        Ad[0:3, 6:9] = Rz * t
        I_g = np.dot(np.dot(Rz, self.I_b), Rz.T)  # inertia matrix in world frame
        I_g_inv = np.linalg.inv(I_g)
        Bd[7:10, 0:3] = I_g_inv @ utils.hat(self.rh) * t
        G = self.G
        Q = self.Q
        R = self.R
        X = cp.Variable((N + 1, n_x))
        U = cp.Variable((N, n_u))
        cost = 0
        constr = []
        U_ref = np.zeros(n_u)
        # --- calculate cost & constraints --- #
        for k in range(0, N):
            kf = 10 if k == N - 1 else 1  # terminal cost
            kuf = 0 if k == N - 1 else 1  # terminal cost
            z = X[k, 5]
            fx = U[k, 0]
            fy = U[k, 1]
            fz = U[k, 2]
            if (k % 2) == 1:  # odd
                U_ref[-1] = 0
                cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :] + G,
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # even
                U_ref[-1] = m * self.g * 2
                cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :] + G,
                           0 >= fx - mu * fz,
                           0 >= -fx - mu * fz,
                           0 >= fy - mu * fz,
                           0 >= -fy - mu * fz,
                           fz >= 0,
                           z <= 3,
                           z >= 0]
        constr += [X[0, :] == X_in, X[N, :] == X_ref[-1, :]]  # initial and final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.SCS)  # , verbose=True)
        u = U.value
        x = X.value
        if u is None:
            raise Exception("\n *** QP FAILED *** \n")
        # breakpoint()
        return u, x
