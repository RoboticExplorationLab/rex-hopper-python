"""
Copyright (C) 2021-2022 Benjamin Bokser
"""
import numpy as np


class Spring:

    def __init__(self, model, **kwargs):
        """
        linear extension spring b/t joints 1 and 3 of parallel mechanism
        """
        self.ks = model["ks"]  # spring constant, N/m
        self.dir_s = model["springpolarity"]
        init_q = model["init_q"]
        self.init_q = [init_q[0], init_q[2]]
        self.L = model["linklengths"]
        L0 = self.L[0]  # .15
        L2 = self.L[2]  # .3
        self.r0 = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(2.5 * np.pi / 180))  # 0.17

    def fn_spring(self, q):
        """
        effect of spring tension approximated by applying torques to joints 0 and 2
        """
        init_q = self.init_q
        k = self.ks
        L0 = self.L[0]
        L2 = self.L[2]
        r0 = self.r0
        if q is None:
            q0 = init_q[0]
            q2 = init_q[2]
        else:
            q0 = q[0] + init_q[0]
            q2 = q[2] + init_q[1]
        gamma = abs(q2 - q0)
        r = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(gamma))  # length of spring
        # if r < r0:
        #     print("error: incorrect spring params, r = ", r, " and r0 = ", r0, "\n gamma = ", gamma)
        T = k * (r - r0)  # spring tension force
        alpha = np.arccos((-L0 ** 2 + L2 ** 2 + r ** 2) / (2 * L2 * r))
        beta = np.arccos((-L2 ** 2 + L0 ** 2 + r ** 2) / (2 * L0 * r))
        tau_s0 = -T * np.sin(beta) * L0
        tau_s1 = T * np.sin(alpha) * L2
        tau_s = np.array([tau_s0, tau_s1]) * self.dir_s
        return tau_s

    def fn_no_spring(self, q):
        # use this if no spring
        return np.zeros(2)
