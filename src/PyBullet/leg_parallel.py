"""
Copyright (C) 2013 Travis DeWolf
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np
import sympy as sym
import csv
import os

import transforms3d

from legbase import LegBase


class Leg(LegBase):

    def __init__(self, l, model, altsize=1, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            init_dq = [0., 0.]

        if init_q is None:
            init_q = [-30 * np.pi / 180, -150 * np.pi / 180]

        self.DOF = 4

        LegBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        self.L = l # np.array([L0, L1, L2, L3, L4])
        model_path = None
        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        if model == 'design':
            if altsize == 1:
                model_path = "res/flyhopper_robot/urdf/flyhopper_robot.csv"
            elif altsize == 0.8:
                model_path = "res/flyhopper_robot_0_8/urdf/flyhopper_robot_0_8.csv"
            elif altsize == 1.2:
                model_path = "res/flyhopper_robot_1_2/urdf/flyhopper_robot_1_2.csv"
            else:
                print("error: invalid size")
                model_path = None
        elif model == 'parallel':
            model_path = "res/flyhopper_parallel/urdf/flyhopper_parallel.csv"
        path = os.path.join(path_parent, os.path.pardir, model_path)
        with open(path, 'r') as csvfile:
            data_direct = csv.reader(csvfile, delimiter=',')
            next(data_direct)  # skip headers
            values_direct = list(zip(*(row for row in data_direct)))  # transpose rows to columns
            values_direct = np.array(values_direct)  # convert list of nested lists to array

        # values = []
        self.inertial_data = False

        if self.inertial_data is True:
            inertia_data = str('path/inertia_data.csv')  # TODO: Add

            with open(inertia_data, 'r') as csvfile:
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
            self.coml = values[7].astype(np.float)

        else:
            self.coml = 0.5*self.L
            ixx = values_direct[8].astype(np.float)
            ixy = values_direct[9].astype(np.float)
            ixz = values_direct[10].astype(np.float)
            iyy = values_direct[11].astype(np.float)
            iyz = values_direct[12].astype(np.float)
            izz = values_direct[13].astype(np.float)

        self.mass = values_direct[7].astype(np.float)
        self.mass = np.delete(self.mass, 0)  # remove body value

        # mass matrices and gravity
        self.MM = []
        self.Fg = []
        self.gravity = np.array([[0, 0, -9.807]]).T
        self.extra = np.array([[0, 0, 0]]).T

        for i in range(0, 2):
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

        self.angles = init_q
        self.q_previous = init_q
        self.dq_previous = init_dq
        self.d2q_previous = init_dq
        self.kv = 0.05
        self.reset()
        self.q_calibration = np.array(init_q)

        #-----------------------#
        L0 = self.L[0]
        L1 = self.L[1]
        l0 = self.coml[0]
        l1 = self.coml[1]
        sym.var('q0 q1')
        T_0_org = sym.Matrix([[sym.cos(q0), 0, -sym.sin(q0), L0 * sym.cos(q0)],
                              [0, 1, 0, 0],
                              [sym.sin(q0), 0, sym.cos(q0), L0 * sym.sin(q0)],
                              [0, 0, 0, 1]])
        T_1_0 = sym.Matrix([[sym.cos(q1), 0, -sym.sin(q1), L1 * sym.cos(q1)],
                            [0, 1, 0, 0],
                            [sym.sin(q1), 0, sym.cos(q1), L1 * sym.sin(q1)],
                            [0, 0, 0, 1]])

        com0 = sym.Matrix([[l0 * sym.cos(q0)],
                           [0],
                           [l0 * sym.sin(q0)],
                           [1]])
        com1 = sym.Matrix([[l1 * sym.cos(q1)],
                           [0],
                           [l1 * sym.sin(q1)],
                           [1]])

        xee = sym.Matrix([[L1*sym.cos(q1)],
                          [0],
                          [L1*sym.sin(q1)],
                          [1]])

        JCOM0 = com0.jacobian([q0, q1])
        JCOM0.row_del(3)

        JCOM0_init = JCOM0.row_insert(4, sym.Matrix([[0, 0],
                                                     [1, 0],
                                                     [0, 0]]))
        self.JCOM0_init = sym.lambdify([q0, q1], JCOM0_init)

        JCOM1 = (T_0_org.multiply(com1)).jacobian([q0, q1])
        JCOM1.row_del(3)
        JCOM1_init = JCOM1.row_insert(4, sym.Matrix([[0, 0],
                                                     [1, 1],
                                                     [0, 0]]))
        self.JCOM1_init = sym.lambdify([q0, q1], JCOM1_init)

        # T_1_org = T_0_org.multiply(T_1_0)
        # JEE_v = (T_1_org.multiply(xee)).jacobian([q0, q1])
        JEE_v = (T_0_org.multiply(xee)).jacobian([q0, q1])
        JEE_v.row_del(3)
        JEE_w = sym.Matrix([[0, 0],
                            [1, 1],
                            [0, 0]])
        JEE_init = JEE_v.row_insert(4, JEE_w)
        self.JEE_init = sym.lambdify([q0, q1], JEE_init)

        #----Rotation------------#
        R_0_org = sym.Matrix([[sym.cos(q0), 0, -sym.sin(q0)],
                              [0, 1, 0],
                              [sym.sin(q0), 0, sym.cos(q0)]])
        R_1_0 = sym.Matrix([[sym.cos(q1), 0, -sym.sin(q1)],
                            [0, 1, 0],
                            [sym.sin(q1), 0, sym.cos(q1)]])
        R_1_org_init = R_0_org.multiply(R_1_0)
        self.R_1_org_init = sym.lambdify([q0, q1], R_1_org_init)

    def gen_jacCOM0(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q
        # JCOM0 = np.zeros((6, 4))
        JCOM0 = self.JCOM0_init(q[0], q[1])
        JCOM0 = np.array(JCOM0).astype(np.float64)
        return JCOM0

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q

        JCOM1 = self.JCOM1_init(q[0], q[1])
        JCOM1 = np.array(JCOM1).astype(np.float64)
        return JCOM1

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from the end effector to the origin frame"""
        q = self.q if q is None else q

        JEE = self.JEE_init(q[0], q[1])
        JEE = np.array(JEE).astype(np.float64)
        return JEE

    def gen_Mq(self, q=None):
        # Mass matrix
        M0 = self.MM[0]
        M1 = self.MM[1]

        JCOM0 = self.gen_jacCOM0(q=q)
        JCOM1 = self.gen_jacCOM1(q=q)

        Mq = (np.dot(JCOM0.T, np.dot(M0, JCOM0)) +
              np.dot(JCOM1.T, np.dot(M1, JCOM1)))

        return Mq

    def gen_grav(self, b_orient, q=None):
        # Generate gravity term g(q)
        body_grav = np.dot(b_orient.T, self.gravity)  # adjust gravity vector based on body orientation
        body_grav = np.append(body_grav, np.array([[0, 0, 0]]).T)
        for i in range(0, 2):
            fgi = float(self.mass[i])*body_grav  # apply mass*gravity
            self.Fg.append(fgi)

        J0T = np.transpose(self.gen_jacCOM0(q=q))
        J1T = np.transpose(self.gen_jacCOM1(q=q))

        gq = J0T.dot(self.Fg[0]) + J1T.dot(self.Fg[1])

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
        # if dq is None:
        #     dq = self.dq
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
        self.dq = (self.q - self.q_previous) / self.dt
        # Make sure this only happens once per time step
        # self.d2q = [i * self.kv for i in self.d2q_previous] + (self.dq - self.dq_previous) / self.dt
        self.d2q = (self.dq - self.dq_previous) / self.dt

        self.q_previous = self.q
        self.dq_previous = self.dq
        self.d2q_previous = self.d2q
