"""
Copyright (C) 2020-2021 Benjamin Bokser
"""

import numpy as np
import sympy as sp
import csv
import os

import transforms3d
from sympy.physics.vector import dynamicsymbols

from legbase import LegBase


class Leg(LegBase):

    def __init__(self, model, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            init_dq = [0., 0.]

        if init_q is None:
            init_q = [-30 * np.pi / 180, -150 * np.pi / 180]

        self.DOF = len(init_q)

        LegBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        self.L = np.array(model["linklengths"])
        csv_path = model["csvpath"]
        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        path = os.path.join(path_parent, os.path.pardir, csv_path)
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
        self.MM = []
        self.Fg = []
        self.I = []
        # self.gravity = np.array([[0, 0, -9.807]]).T
        g = 9.807
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
            self.MM.append(M)
            self.I.append(iyy[i])  # all rotations are in y axis

        self.angles = init_q
        self.q_previous = init_q
        self.dq_previous = init_dq
        self.d2q_previous = init_dq
        # self.kv = 0.05
        self.reset()
        self.q_calibration = np.array(init_q)

        # --- Forward Kinematics --- #
        # we're pretending y and z are switched, just roll with it
        l0 = self.L[0]
        l1 = self.L[1]
        l2 = self.L[2]
        l3 = self.L[3]
        l4 = self.L[4]
        l5 = self.L[5]
        lee = np.sqrt((l3+l4)**2 + l5**2)
        l_c0x = self.coml[0, 0]
        l_c0y = self.coml[2, 0]
        l_c1 = self.coml[0, 1]
        l_c2 = self.coml[0, 2]
        l_ceex = self.coml[0, 3]
        l_ceey = self.coml[2, 3]

        l_c0 = np.sqrt(l_c0x**2 + l_c0y**2)
        alpha0 = np.arctan2(l_c0y, l_c0x)
        l_cee = np.sqrt(l_ceex**2 + l_ceey**2)
        alpha3 = np.arctan2(l_ceey, l_ceex)
        alphaee = np.arctan2(l5, l3 + l4)

        m0 = self.mass[0]
        m1 = self.mass[1]
        m2 = self.mass[2]
        m3 = self.mass[3]

        I0 = self.I[0]
        I1 = self.I[1]
        I2 = self.I[2]
        I3 = self.I[3]

        # sp.var('q0 q1 q2 q3')
        q0 = dynamicsymbols('q0')
        q1 = dynamicsymbols('q1')
        q2 = dynamicsymbols('q2')
        q3 = dynamicsymbols('q3')
        q0d = dynamicsymbols('q0d')
        q1d = dynamicsymbols('q1d')
        q2d = dynamicsymbols('q2d')
        q3d = dynamicsymbols('q3d')
        q0dd = dynamicsymbols('q0dd')
        q1dd = dynamicsymbols('q1dd')
        q2dd = dynamicsymbols('q2dd')
        q3dd = dynamicsymbols('q3dd')

        t = sp.Symbol('t')

        x0 = l_c0 * sp.cos(q0 + alpha0)
        y0 = l_c0 * sp.sin(q0 + alpha0)
        x1 = l0 * sp.cos(q0) + l_c1 * sp.cos(q0 + q1)
        y1 = l0 * sp.sin(q0) + l_c1 * sp.sin(q0 + q1)

        x2 = l_c2 * sp.cos(q2)
        y2 = l_c2 * sp.sin(q2)
        x3 = l2 * sp.cos(q2) + l_cee * sp.cos(q2 + q3 + alpha3)
        y3 = l2 * sp.sin(q2) + l_cee * sp.sin(q2 + q3 + alpha3)

        x0d = sp.diff(x0, t)
        y0d = sp.diff(y0, t)
        x1d = sp.diff(x1, t)
        y1d = sp.diff(y1, t)
        x2d = sp.diff(x2, t)
        y2d = sp.diff(y2, t)
        x3d = sp.diff(x3, t)
        y3d = sp.diff(y3, t)

        U0 = m0 * g * y0
        U1 = m1 * g * y1
        U2 = m2 * g * y2
        U3 = m3 * g * y3

        v0_squared = x0d**2 + y0d**2
        v1_squared = x1d**2 + y1d**2
        v2_squared = x2d**2 + y2d**2
        v3_squared = x3d**2 + y3d**2
        T0 = 0.5 * m0 * v0_squared + 0.5 * I0 * q0d**2
        T1 = 0.5 * m1 * v1_squared + 0.5 * I1 * q1d**2
        T2 = 0.5 * m2 * v2_squared + 0.5 * I2 * q2d**2
        T3 = 0.5 * m3 * v3_squared + 0.5 * I3 * q3d**2

        U = U0 + U1 + U2 + U3
        T = T0 + T1 + T2 + T3

        # Le Lagrangian
        L = sp.trigsimp(T - U)
        L = L.subs(sp.Derivative(q0, t), q0d)  # substitute d/dt q2 with q2d
        L = L.subs(sp.Derivative(q1, t), q1d)  # substitute d/dt q1 with q1d
        L = L.subs(sp.Derivative(q2, t), q2d)  # substitute d/dt q2 with q2d
        L = L.subs(sp.Derivative(q3, t), q3d)  # substitute d/dt q2 with q2d

        # Lagrange-Euler Equation
        LE0 = sp.diff(sp.diff(L, q0d), t) - sp.diff(L, q0)
        LE1 = sp.diff(sp.diff(L, q1d), t) - sp.diff(L, q1)
        LE2 = sp.diff(sp.diff(L, q2d), t) - sp.diff(L, q2)
        LE3 = sp.diff(sp.diff(L, q3d), t) - sp.diff(L, q3)
        LE = sp.Matrix([LE0, LE1, LE2, LE3])

        # subs first derivative
        LE = LE.subs(sp.Derivative(q0, t), q0d)  # substitute d/dt q1 with q1d
        LE = LE.subs(sp.Derivative(q1, t), q1d)  # substitute d/dt q1 with q1d
        LE = LE.subs(sp.Derivative(q2, t), q2d)  # substitute d/dt q2 with q2d
        LE = LE.subs(sp.Derivative(q3, t), q3d)  # substitute d/dt q1 with q1d
        # subs second derivative
        LE = LE.subs(sp.Derivative(q0d, t), q0dd)  # substitute d/dt q1d with q1dd
        LE = LE.subs(sp.Derivative(q1d, t), q1dd)  # substitute d/dt q1d with q1dd
        LE = LE.subs(sp.Derivative(q2d, t), q2dd)  # substitute d/dt q2d with q2dd
        LE = LE.subs(sp.Derivative(q3d, t), q3dd)  # substitute d/dt q1d with q1dd
        LE = sp.expand(sp.simplify(LE))

        # Generalized mass matrix
        M = sp.zeros(4, 4)
        M[0, 0] = sp.collect(LE[0], q0dd).coeff(q0dd)
        M[0, 1] = sp.collect(LE[0], q1dd).coeff(q1dd)
        M[0, 2] = sp.collect(LE[0], q2dd).coeff(q2dd)
        M[0, 3] = sp.collect(LE[0], q3dd).coeff(q3dd)
        M[1, 0] = sp.collect(LE[1], q0dd).coeff(q0dd)
        M[1, 1] = sp.collect(LE[1], q1dd).coeff(q1dd)
        M[1, 2] = sp.collect(LE[1], q2dd).coeff(q2dd)
        M[1, 3] = sp.collect(LE[1], q3dd).coeff(q3dd)
        M[2, 0] = sp.collect(LE[2], q0dd).coeff(q0dd)
        M[2, 1] = sp.collect(LE[2], q1dd).coeff(q1dd)
        M[2, 2] = sp.collect(LE[2], q2dd).coeff(q2dd)
        M[2, 3] = sp.collect(LE[2], q3dd).coeff(q3dd)
        M[3, 0] = sp.collect(LE[3], q0dd).coeff(q0dd)
        M[3, 1] = sp.collect(LE[3], q1dd).coeff(q1dd)
        M[3, 2] = sp.collect(LE[3], q2dd).coeff(q2dd)
        M[3, 3] = sp.collect(LE[3], q3dd).coeff(q3dd)
        self.M_init = sp.lambdify([q0, q1, q2, q3], M)

        # Gravity Matrix
        G = LE
        G = G.subs(q0d, 0)
        G = G.subs(q1d, 0)  # must remove q derivative terms manually
        G = G.subs(q2d, 0)
        G = G.subs(q3d, 0)
        G = G.subs(q0dd, 0)
        G = G.subs(q1dd, 0)
        G = G.subs(q2dd, 0)
        G = G.subs(q3dd, 0)
        self.G_init = sp.lambdify([q0, q1, q2, q3], G)

        # Coriolis Matrix
        # assume anything without qdd minus G is C
        C = LE
        C = C.subs(q0dd, 0)
        C = C.subs(q1dd, 0)
        C = C.subs(q2dd, 0)
        C = C.subs(q3dd, 0)
        C = C - G
        self.C_init = sp.lambdify([q0, q1, q2, q3], C)

        # --- Jacobians --- #


    def gen_jacCOM0(self, q=None):
        q = self.q if q is None else q
        JCOM0 = self.JCOM0_init(q[0], q[1], q[2], q[3])
        JCOM0 = np.array(JCOM0).astype(np.float64)
        return JCOM0

    def gen_jacCOM1(self, q=None):
        q = self.q if q is None else q
        JCOM1 = self.JCOM1_init(q[0], q[1], q[2], q[3])
        JCOM1 = np.array(JCOM1).astype(np.float64)
        return JCOM1

    def gen_jacCOM2(self, q=None):
        q = self.q if q is None else q
        JCOM2 = self.JCOM2_init(q[0], q[1], q[2], q[3])
        JCOM2 = np.array(JCOM2).astype(np.float64)
        return JCOM2

    def gen_jacCOM3(self, q=None):
        q = self.q if q is None else q
        JCOM3 = self.JCOM3_init(q[0], q[1], q[2], q[3])
        JCOM3 = np.array(JCOM3).astype(np.float64)
        return JCOM3

    def gen_jacEE(self, q=None):
        q = self.q if q is None else q
        JEE = self.JEE_init(q[0], q[1], q[2], q[3])
        JEE = np.array(JEE).astype(np.float64)
        return JEE

    def gen_D(self, q=None):
        q = self.q if q is None else q
        D = self.D_init(q[0], q[1], q[2], q[3])
        D = np.array(D).astype(np.float64)
        return D

    def gen_Mq(self, q=None):
        # Mass matrix
        M0 = self.MM[0]
        M1 = self.MM[1]
        M2 = self.MM[2]
        M3 = self.MM[3]

        JCOM0 = self.gen_jacCOM0(q=q)
        JCOM1 = self.gen_jacCOM1(q=q)
        JCOM2 = self.gen_jacCOM2(q=q)
        JCOM3 = self.gen_jacCOM3(q=q)

        Mq = (np.dot(JCOM0.T, np.dot(M0, JCOM0)) +
              np.dot(JCOM1.T, np.dot(M1, JCOM1)) +
              np.dot(JCOM2.T, np.dot(M2, JCOM2)) +
              np.dot(JCOM3.T, np.dot(M3, JCOM3)))

        return Mq

    def gen_grav(self, b_orient, q=None):
        # Generate gravity term g(q)
        body_grav = np.dot(b_orient.T, self.gravity)  # adjust gravity vector based on body orientation
        body_grav = np.append(body_grav, np.array([[0, 0, 0]]).T)
        for i in range(0, 4):
            fgi = float(self.mass[i])*body_grav  # apply mass*gravity
            self.Fg.append(fgi)

        J0T = np.transpose(self.gen_jacCOM0(q=q))
        J1T = np.transpose(self.gen_jacCOM1(q=q))
        J2T = np.transpose(self.gen_jacCOM2(q=q))
        J3T = np.transpose(self.gen_jacCOM3(q=q))

        gq = J0T.dot(self.Fg[0]) + J1T.dot(self.Fg[1]) + J2T.dot(self.Fg[2]) + J3T.dot(self.Fg[3])

        return gq.reshape(-1, )

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
        """forward kinematics
        Compute x,y,z position of end effector relative to base.
        This outputs four sets of xyz values, one for each joint including the end effector.

        q np.array: a set of angles to return positions for
        """
        if q is None:
            q0 = self.q[0]
            q2 = self.q[1]
        else:
            q0 = q[0]
            q2 = q[1]

        L0 = self.L[0]
        L1 = self.L[1]
        L2 = self.L[2]
        L3 = self.L[3]
        L4 = self.L[4]
        L5 = self.L[5]
        d = 0

        x0 = L0 * np.cos(q0)
        y0 = L0 * np.sin(q0)
        rho = np.sqrt((x0 + d) ** 2 + y0 ** 2)

        # This works to calculate h as well, but might be slightly slower bc more trig
        # x1 = L2 * np.cos(q2)
        # y1 = L2 * np.sin(q2)
        # h = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

        gamma = abs(q2 - q0)
        h = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(gamma))  # length of spring
        mu = np.arccos((L3 ** 2 + h ** 2 - L1 ** 2) / (2 * L3 * h))
        eta = np.arccos((h**2 + L2**2 - rho**2)/(2*h*L2))
        alpha = np.pi - (eta + mu) + q2
        x = L2 * np.cos(q2) + (L3 + L4) * np.cos(alpha) - d + L5 * np.cos(alpha - np.pi/2)
        y = 0
        z = L2 * np.sin(q2) + (L3 + L4) * np.sin(alpha) + L5 * np.cos(alpha - np.pi/2)

        return np.array([x, y, z], dtype=float)

    def velocity(self, q=None):  # dq=None
        # Calculate operational space linear velocity vector
        q = self.q if q is None else q
        JEE = self.gen_jacEE(q=q)
        return np.dot(JEE, self.dq).flatten()

    def orientation(self, b_orient, q=None):
        # Calculate orientation of end effector in quaternions
        q = self.q if q is None else q
        # REE = np.zeros((3, 3))  # rotation matrix
        REE = self.R_1_org_init(q[0], q[1])
        REE = np.array(REE).astype(np.float64)
        REE = np.dot(b_orient, REE)
        q_e = transforms3d.quaternions.mat2quat(REE)
        q_e = q_e / np.linalg.norm(q_e)  # convert to unit vector quaternion

        return q_e

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