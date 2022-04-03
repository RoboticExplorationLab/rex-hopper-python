"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm


class Mpc:

    def __init__(self, t, N, m, g, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.m = m  # kg
        self.A = np.array([[0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, -1],
                           [0, 0, 0, 0, 0, 0, 0]])
        self.B = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [1 / self.m, 0, 0],
                           [0, 1 / self.m, 0],
                           [0, 0, 1 / self.m],
                           [0, 0, 0]])
        self.N = N  # prediction horizon
        self.mu = mu  # coefficient of friction
        self.g = g
        n_x = np.shape(self.A)[1]
        n_u = np.shape(self.B)[1]
        AB = np.vstack((np.hstack((self.A, self.B)), np.zeros((n_u, n_x + n_u))))
        M = expm(AB * t)
        self.Ad = M[0:n_x, 0:n_x]
        self.Bd = M[0:n_x, n_x:n_x + n_u]
        self.Q = np.eye(n_x)
        self.R = np.eye(n_u)
        np.fill_diagonal(self.Q, [1., 1., 1., 0.01, 0.01, 0.01, 0.])
        np.fill_diagonal(self.R, [0., 0., 0.])
        self.n_u = n_u
        self.n_x = n_x

    def mpcontrol(self, X_in, X_ref):
        N = self.N
        m = self.m
        mu = self.mu
        n_x = self.n_x
        n_u = self.n_u
        Ad = self.Ad
        Bd = self.Bd
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
            z = X[k, 2]
            fx = U[k, 0]
            fy = U[k, 1]
            fz = U[k, 2]
            if ((k + 1) % 2) == 0:  # even
                U_ref[-1] = 0
                cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :],
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # odd
                U_ref[-1] = m * self.g * 2
                cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :],
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
