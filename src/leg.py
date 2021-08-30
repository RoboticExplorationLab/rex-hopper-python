"""
Copyright (C) 2013 Travis DeWolf
Copyright (C) 2020 Benjamin Bokser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import sympy as sym
import csv

import transforms3d

from legbase import LegBase


class Leg(LegBase):

    def __init__(self, init_q=None, init_dq=None, **kwargs):

        if init_dq is None:
            init_dq = [0., 0.]

        if init_q is None:
            init_q = [-150 * np.pi / 180,
                      120 * np.pi / 180]

        self.DOF = 4

        LegBase.__init__(self, init_q=init_q, init_dq=init_dq, **kwargs)

        # link lengths (mm) must be manually updated
        L0 = .3
        L1 = .3
        self.L = np.array([L0, L1])

        with open('res/flyhopper_mockup/urdf/flyhopper_mockup.csv', 'r') as csvfile:
            data_direct = csv.reader(csvfile, delimiter=',')
            next(data_direct)  # skip headers
            values_direct = list(zip(*(row for row in data_direct)))  # transpose rows to columns
            values_direct = np.array(values_direct)  # convert list of nested lists to array

        # values = []
        self.inertial_data = False

        if self.inertial_data is True:
            inertia_data = str('path/inertia_data.csv')

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
        '''
        xee = sym.Matrix([[0],  # L1*sym.cos(q1)],
                          [0],
                          [0],  # L1*sym.sin(q1)],
                          [1]])
        '''
        xee = sym.Matrix([[L1*sym.cos(q1)],
                          [0],
                          [L1*sym.sin(q1)],
                          [1]])

        JCOM0 = com0.jacobian([q0, q1])
        JCOM0.row_del(3)
        self.JCOM0_init = JCOM0.row_insert(4, sym.Matrix([[0, 0],
                                                          [1, 0],
                                                          [0, 0]]))

        JCOM1 = (T_0_org.multiply(com1)).jacobian([q0, q1])
        JCOM1.row_del(3)
        self.JCOM1_init = JCOM1.row_insert(4, sym.Matrix([[0, 0],
                                                          [1, 1],
                                                          [0, 0]]))

        # T_1_org = T_0_org.multiply(T_1_0)
        # JEE_v = (T_1_org.multiply(xee)).jacobian([q0, q1])
        JEE_v = (T_0_org.multiply(xee)).jacobian([q0, q1])
        JEE_v.row_del(3)
        JEE_w = sym.Matrix([[0, 0],
                            [1, 1],
                            [0, 0]])
        self.JEE_init = JEE_v.row_insert(4, JEE_w)

        #----Rotation------------#
        R_0_org = sym.Matrix([[sym.cos(q0), 0, -sym.sin(q0)],
                              [0, 1, 0],
                              [sym.sin(q0), 0, sym.cos(q0)]])
        R_1_0 = sym.Matrix([[sym.cos(q1), 0, -sym.sin(q1)],
                            [0, 1, 0],
                            [sym.sin(q1), 0, sym.cos(q1)]])
        self.R_1_org_init = R_0_org.multiply(R_1_0)

    def gen_jacCOM0(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q
        # JCOM0 = np.zeros((6, 4))
        JCOM0 = self.JCOM0_init.subs({q0: q[0], q1: q[1]})
        JCOM0 = np.array(JCOM0).astype(np.float64)
        return JCOM0

    def gen_jacCOM1(self, q=None):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q = self.q if q is None else q

        JCOM1 = self.JCOM1_init.subs({q0: q[0], q1: q[1]})
        JCOM1 = np.array(JCOM1).astype(np.float64)
        return JCOM1

    def gen_jacEE(self, q=None):
        """Generates the Jacobian from the end effector to the origin frame"""
        q = self.q if q is None else q

        JEE = self.JEE_init.subs({q0: q[0], q1: q[1]})
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

        x = xyz[0]
        # y = xyz[1]
        z = xyz[2]

        q1 = np.arccos((- L0**2 - L1**2 + x**2 + z**2)/(2*L0*L1))
        beta = np.arctan2(L1*np.sin(q1), L0 + L1*np.cos(q1))
        gamma = np.arctan2(z, x)  # TODO: Check the sign on this
        q0 = -(np.pi + gamma + beta)

        return np.array([q0, q1], dtype=float)

    def position(self, q=None):
        """forward kinematics
        Compute x,y,z position of end effector relative to base.
        This outputs four sets of xyz values, one for each joint including the end effector.

        q np.array: a set of angles to return positions for
        """
        if q is None:
            q0 = self.q[0]
            q1 = self.q[1]
        else:
            q0 = q[0]
            q1 = q[1]

        L0 = self.L[0]
        L1 = self.L[1]

        x = L0 * np.cos(q0) + L1 * np.cos(q0 + q1)

        y = 0

        z = L0 * np.sin(q0) + L1 * np.sin(q0 + q1)

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
        REE = self.R_1_org_init.subs({q0: q[0], q1: q[1]})
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
