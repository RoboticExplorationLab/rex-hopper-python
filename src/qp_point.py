"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import expm


class Qp:

    def __init__(self, t, g, N, m, mu, **kwargs):
        self.t = t  # sampling time (s)
        self.N = N  # prediction horizon
        self.m = m  # kg
        self.mu = mu  # coefficient of friction
        self.g = g
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1 / self.m, 0, 0],
                      [0, 1 / self.m, 0],
                      [0, 0, 1 / self.m]])
        G = np.array([[0, 0, 0, 0, 0, -g]]).T
        n_x = np.shape(A)[1]
        n_u = np.shape(B)[1]
        ABG = np.hstack((A, B, G))
        ABG.resize((n_x + n_u + 1, n_x + n_u + 1))
        M = expm(ABG * t)
        self.Ad = M[0:n_x, 0:n_x]
        self.Bd = M[0:n_x, n_x:n_x+n_u]
        self.Gd = M[0:n_x, -1]
        self.Q = np.eye(n_x)
        self.R = np.eye(n_u)
        self.n_x = n_x
        self.n_u = n_u
        np.fill_diagonal(self.Q, [1., 1., 15., 0.1, 0.1, 0.1])
        np.fill_diagonal(self.R, [0., 0., 0.])

    def qpcontrol(self, X_in, X_ref, Ck):
        N = self.N
        m = self.m
        mu = self.mu
        Ad = self.Ad
        Bd = self.Bd
        Gd = self.Gd
        g = self.g
        n_x = self.n_x
        n_u = self.n_u
        Q = self.Q
        R = self.R
        X = cp.Variable((N+1, n_x))
        U = cp.Variable((N, n_u))
        cost = 0
        constr = []

        # --- calculate cost & constraints --- #
        for k in range(0, N):
            kf = 10 if k == N - 1 else 1  # terminal cost
            kuf = 0 if k == N - 1 else 1  # terminal cost
            z = X[k, 2]
            fx = U[k, 0]
            fy = U[k, 1]
            fz = U[k, 2]
            if Ck[k] == 0:
                U_ref = np.zeros(n_u)
                cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :] + Gd,
                           0 == fx,
                           0 == fy,
                           0 == fz]
            else:  # odd
                U_ref = np.array([10, 10, m * g])
                cost += cp.quad_form(X[k + 1, :] - X_ref[k, :], Q * kf) + cp.quad_form(U[k, :] - U_ref, R * kuf)
                constr += [X[k + 1, :] == Ad @ X[k, :] + Bd @ U[k, :] + Gd,
                           0 >= fx - mu * fz,
                           0 >= -fx - mu * fz,
                           0 >= fy - mu * fz,
                           0 >= -fy - mu * fz,
                           fz >= 0,
                           z >= 0.2,
                           z <= 1]
        constr += [X[0, :] == X_in]  # initial condition
        # constr += [X[N, :] == X_ref[-1, :]]  # final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  #, verbose=True)
        u = U.value
        x = X.value
        if u is None:
            raise Exception("\n *** QP FAILED *** \n")
        # print(u)
        # breakpoint()
        return u, x
