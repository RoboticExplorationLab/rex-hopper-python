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
        self.Q[n_u, n_u] *= 0.01
        self.Q[n_u + 1, n_u + 1] *= 0.01
        self.Q[n_u + 2, n_u + 2] *= 0.01
        self.R = np.eye(n_u) * 0
        self.n_u = n_u
        self.n_x = n_x

    def mpcontrol(self, X_in, X_ref, s):
        N = self.N
        m = self.m
        mu = self.mu
        n_x = self.n_x
        n_u = self.n_u
        Ad = self.Ad
        Bd = self.Bd
        Q = self.Q
        R = self.R
        X = cp.Variable((n_x, N + 1))
        U = cp.Variable((n_u, N))
        cost = 0
        constr = []
        U_ref = np.zeros(n_u)
        U_ref[-1] = m * self.g
        # --- calculate cost & constraints --- #
        for k in range(0, N):
            kf = 10 if k == N - 1 else 1  # terminal cost
            kuf = 0 if k == N - 1 else 1  # terminal cost
            cost += cp.quad_form(X[:, k+1] - X_ref[k, :], Q * kf) + cp.quad_form(U[:, k] - U_ref, R * kuf)
            fx = U[0, k]
            fy = U[1, k]
            fz = U[2, k]
            if ((k + s) % 2) == 0:  # even
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                           0 == fx,  # fx
                           0 == fy,  # fy
                           0 == fz]  # fz
            else:  # odd
                constr += [X[:, k + 1] == Ad @ X[:, k] + Bd @ U[:, k],
                           0 >= fx - mu * fz,
                           0 >= -fx - mu * fz,
                           0 >= fy - mu * fz,
                           0 >= -fy - mu * fz,
                           0 <= fz]
        constr += [X[:, 0] == X_in, X[:, N] == X_ref[-1, :]]  # initial and final condition
        # --- set up solver --- #
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.OSQP)  # , verbose=True)
        u = np.zeros((n_u, N)) if U.value is None else U.value
        # print(X.value)
        # breakpoint()
        return u, (s % 2)
