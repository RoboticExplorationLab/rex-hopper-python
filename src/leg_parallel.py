"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import sympy as sp
import csv
import os
import pickle
import transforms3d

import calc_parallel

class Leg:

    def __init__(self, dt, model, recalc, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            init_dq = [0., 0., 0., 0.]

        if init_q is None:
            init_q = [-30 * np.pi / 180, -120 * np.pi / 180, -150 * np.pi / 180, 120 * np.pi / 180]

        self.q = init_q
        self.dq = init_dq
        self.DOF = len(init_q)
        self.singularity_thresh = 0.00025
        self.dt = dt
        self.L = np.array(model["linklengths"])
        csv_path = model["csvpath"]
        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        # path = os.path.join(path_parent, os.path.pardir, csv_path)
        path = os.path.join(path_parent, csv_path)
        with open(path, 'r') as csvfile:
            data_direct = csv.reader(csvfile, delimiter=',')
            next(data_direct)  # skip headers
            values_direct = list(zip(*(row for row in data_direct)))  # transpose rows to columns
            values_direct = np.array(values_direct)  # convert list of nested lists to array

        ixx = values_direct[8].astype(np.float)
        ixy = values_direct[9].astype(np.float)
        ixz = values_direct[10].astype(np.float)
        iyy = values_direct[11].astype(np.float)
        iyz = values_direct[12].astype(np.float)
        izz = values_direct[13].astype(np.float)

        self.mass = values_direct[7].astype(np.float)
        self.mass = np.delete(self.mass, 0)  # remove body value
        self.coml = values_direct[1:4].astype(np.float)
        self.coml = np.delete(self.coml, 0, axis=1)  # remove body value

        # mass matrices and gravity
        self.II = []
        self.Fg = []
        self.I = []

        num_links = 4
        for i in range(0, num_links):
            M = np.zeros((6, 6))
            M[0:3, 0:3] = np.eye(3) * float(self.mass[i])
            M[3, 3] = ixx[i]
            M[3, 4] = ixy[i]
            M[3, 5] = ixz[i]
            M[4, 3] = ixy[i]
            M[4, 4] = iyy[i]
            M[4, 5] = iyz[i]
            M[5, 3] = ixz[i]
            M[5, 4] = iyz[i]
            M[5, 5] = izz[i]
            self.II.append(M)
            self.I.append(iyy[i])  # all rotations are in y axis

        self.angles = init_q
        self.q_previous = init_q
        self.dq_previous = init_dq
        self.d2q_previous = init_dq

        self.reset()
        self.q_calibration = np.array(init_q)

        self.g = np.array([[0, 0, 9.807]]).T
        # self.g = sp.Matrix([[0, 0, 9.807]]).T  # negative or positive?

        if recalc == True:
            # if recalc is true then recalculate the data and rewrite the pickle
            self.M_init, self.G_init, self.C_init, self.Jf_init, self.df_init, \
            self.D_init, self.d_init, self.cdot_init, self.pos_init, self.Ja_init, \
            self.da_init = calc_parallel.calculate(L=self.L, mass=self.mass, I=self.I, coml=self.coml)
        else:
            # if not true then just open the pickle jar
            pik = "pickle.dat"
            with open(pik, "rb") as f:
                self.M_init, self.G_init, self.C_init, self.Jf_init, self.df_init, \
                self.D_init, self.d_init, self.cdot_init, self.pos_init, self.Ja_init, \
                self.da_init = pickle.load(f)

    def gen_M(self, q=None):
        q = self.q if q is None else q
        M = self.M_init(q[0], q[1], q[2], q[3])
        M = np.array(M).astype(np.float64)
        return M

    def gen_C(self, q=None, dq=None):
        q = self.q if q is None else q
        dq = self.dq if dq is None else dq
        C = self.C_init(q[0], q[1], q[2], q[3], dq[0], dq[1], dq[2], dq[3])
        C = np.array(C).astype(np.float64)
        return C

    def gen_G(self, q=None, g=None):
        q = self.q if q is None else q
        g = self.g if g is None else g
        gx = g[0]
        gy = g[1]
        gz = g[2]
        G = self.G_init(q[0], q[1], q[2], q[3], gx, gy, gz)
        G = np.array(G).astype(np.float64).reshape(-1, 1)
        return G

    def gen_jacF(self, q=None):
        # Full Jacobian
        q = self.q if q is None else q
        Jf = self.Jf_init(q[0], q[1], q[2], q[3])
        Jf = np.array(Jf).astype(np.float64)
        return Jf

    def gen_df(self, q=None, dq=None):
        # del/delq(Jf(q)q_dot)q_dot
        q = self.q if q is None else q
        dq = self.dq if dq is None else dq
        dee = self.df_init(q[0], q[1], q[2], q[3], dq[0], dq[1], dq[2], dq[3])
        dee = np.array(dee).astype(np.float64)
        return dee

    def gen_D(self, q=None):
        # Constraint Jacobian
        q = self.q if q is None else q
        D = self.D_init(q[0], q[1], q[2], q[3])
        D = np.array(D).astype(np.float64)
        return D

    def gen_d(self, q=None, dq=None):
        # del/delq(D(q)q_dot)q_dot
        q = self.q if q is None else q
        dq = self.dq if dq is None else dq
        d = self.d_init(q[0], q[1], q[2], q[3], dq[0], dq[1], dq[2], dq[3])
        d = np.array(d).astype(np.float64)
        return d

    def gen_cdot(self, q=None, dq=None):
        # Constraint Jacobian derivative
        q = self.q if q is None else q
        dq = self.dq if dq is None else dq
        cdot = self.cdot_init(q[0], q[1], q[2], q[3], dq[0], dq[1], dq[2], dq[3])
        cdot = np.array(cdot).astype(np.float64)
        return cdot

    def gen_jacA(self, q=None):
        # End Effector Jacobian (closed chain, actuators only)
        q = self.q if q is None else q
        JA = self.Ja_init(q[0], q[2])
        JA = np.array(JA).astype(np.float64)
        return JA

    def gen_da(self, q=None, dq=None):
        # del/delq(Ja(q)q_dot)q_dot
        q = self.q if q is None else q
        dq = self.dq if dq is None else dq
        da = self.da_init(q[0], q[2], dq[0], dq[2])
        da = np.array(da).astype(np.float64)
        return da

    def gen_Mx(self, J=None, q=None, M = None, **kwargs):
        # Generate the mass matrix in operational space
        if q is None:
            q = self.q
        if M is None:
            M = self.gen_M(q=q)

        if J is None:
            J = np.zeros((3, 4))
            Ja = self.gen_jacA(q=q)
            J[:, 0] = Ja[:, 0]
            J[:, 2] = Ja[:, 1]

        Mx_inv = np.dot(J, np.dot(np.linalg.inv(M), J.T))
        u, s, v = np.linalg.svd(Mx_inv)
        # cut off any singular values that could cause control problems
        for i in range(len(s)):
            s[i] = 0 if s[i] < self.singularity_thresh else 1. / float(s[i])
        # numpy returns U,S,V.T, so have to transpose both here
        Mx = np.dot(v.T, np.dot(np.diag(s), u.T))

        return Mx

    def inv_kinematics(self, xyz):
        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        L4 = self.L[4]
        L5 = self.L[5]
        d = 0  # distance along x b/t motors, 0 for 4-bar link

        x = xyz[0]
        # y = xyz[1]
        z = xyz[2]
        zeta = np.arctan(L5/(L3 + L4))
        rho = np.sqrt(L5**2 + (L3 + L4)**2)
        phi = np.arccos((L2**2 + rho**2 - (x + d)**2 - z**2)/(2*L2*rho)) - zeta
        r1 = np.sqrt((x+d)**2 + z**2)
        ksi = np.arctan2(z, (x+d))
        epsilon = np.arccos((r1**2 + L2**2 - rho**2)/(2*r1*L2))
        q2 = ksi - epsilon
        # print((phi - np.pi - q2)*180/np.pi)
        xm = L2 * np.cos(q2) + L3 * np.cos(phi - np.pi - q2) - d
        zm = L2 * np.sin(q2) - L3 * np.sin(phi - np.pi - q2)
        r2 = np.sqrt(xm**2 + zm**2)
        sigma = np.arccos((-L1**2 + r2**2 + L0**2)/(2*r2*L0))
        q0 = np.arctan2(zm, xm) + sigma
        # print(np.array([q0, q2])*180/np.pi)
        return np.array([q0, q2], dtype=float)

    def position(self, q=None):
        # forward kinematics
        q = self.q if q is None else q
        pos = self.pos_init(q[0], q[2])
        return pos

    def velocity(self, q=None):  # dq=None
        # Calculate operational space linear velocity vector
        q = self.q if q is None else q
        Ja = self.gen_jacA(q=q)
        return np.dot(Ja, self.dq).flatten()

    def reset(self, q=None, dq=None):
        if q is None:
            q = []
        if dq is None:
            dq = []
        if isinstance(q, np.ndarray):
            q = q.tolist()
        if isinstance(dq, np.ndarray):
            dq = dq.tolist()

        if q:
            assert len(q) == self.DOF
        if dq:
            assert len(dq) == self.DOF

    def update_state(self, q_in):
        # Update the local variables
        # Pull values in from simulator and calibrate encoders
        self.q = np.add(q_in.flatten(), self.q_calibration)
        # self.dq = np.reshape([j[1] for j in p.getJointStates(1, range(0, 4))], (-1, 1))
        # self.dq = [i * self.kv for i in self.dq_previous] + (self.q - self.q_previous) / self.dt
        self.dq = (self.q - self.q_previous) / self.dt  # TODO: upgrade from Euler to rk4 or something
        # Make sure this only happens once per time step
        # self.d2q = [i * self.kv for i in self.d2q_previous] + (self.dq - self.dq_previous) / self.dt
        self.d2q = (self.dq - self.dq_previous) / self.dt

        self.q_previous = self.q
        self.dq_previous = self.dq
        self.d2q_previous = self.d2q
