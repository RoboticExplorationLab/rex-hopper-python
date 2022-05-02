"""
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np
import pybullet as p
import pybullet_data
import os
import actuator
import actuator_param
from utils import Z, Q_inv
useRealTime = 0  # Do NOT change to real time


def reaction(numJoints, bot):  # returns joint reaction force
    reaction = np.array([j[2] for j in p.getJointStates(bot, range(numJoints))])  # 4x6 array [Fx, Fy, Fz, Mx, My, Mz]
    forces = reaction[:, 0:3]  # selected all joints [Fx, Fy, Fz]
    torques = reaction[:, 5]
    return forces, torques  # f = np.linalg.norm(reaction[:, 0:3], axis=1)  # magnitude of F


class Sim:

    def __init__(self, X_0, model, spring, dt=1e-3, g=9.807, fixed=False, spr=False,
                 record=False, scale=1, gravoff=False, direct=False):
        self.dt = dt
        self.record_rt = record  # record video in real time
        self.spr = spr
        self.L = model["linklengths"]
        self.model = model["model"]
        self.n_a = model["n_a"]
        self.S = model["S"]
        self.spring_fn = spring.fn_spring if spr is True else spring.fn_no_spring
        self.actuator_q0 = actuator.Actuator(dt=dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1 = actuator.Actuator(dt=dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz = actuator.Actuator(dt=dt, model=actuator_param.actuator_8318)  # r80kv110

        if gravoff == True:
            GRAVITY = 0
        else:
            GRAVITY = -g

        if direct is True:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        model_path = model["urdfpath"]
        # self.bot = p.loadURDF(os.path.join(path_parent, os.path.pardir, model_path), [0, 0, 0.7 * scale],
        self.bot = p.loadURDF(os.path.join(path_parent, model_path), X_0[0:3],  # 0.31
                         robotStartOrientation, useFixedBase=fixed, globalScaling=scale,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])
        self.jointArray = range(p.getNumJoints(self.bot))
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.dt)
        self.numJoints = p.getNumJoints(self.bot)

        p.setRealTimeSimulation(useRealTime)

        self.c_link = 1  # contact link

        # if self.model != 'design_rw' and self.model != 'design_cmg':
        #     vert = p.createConstraint(self.bot, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 0])

        # p.createConstraint(self.bot, 3, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [-0.135, 0, 0], [0, 0, 0])
        jconn_1 = [x * scale for x in [0.135, 0, 0]]
        jconn_2 = [x * scale for x in [-0.0014381, 0, 0.01485326948]]
        linkjoint = p.createConstraint(self.bot, 1, self.bot, 3, p.JOINT_POINT2POINT, [0, 0, 0], jconn_1, jconn_2)
        p.changeConstraint(linkjoint, maxForce=1000)
        self.c_link = 3

        # increase friction of toe to ideal
        # p.changeDynamics(self.bot, self.c_link, lateralFriction=2, contactStiffness=100000, contactDamping=10000)
        p.changeDynamics(self.bot, self.c_link, lateralFriction=3)  # , restitution=0.01)

        # Record Video in real time
        if self.record_rt is True:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        for i in range(self.numJoints):
            # Disable the default velocity/position motor:
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.setJointMotorControl2(self.bot, i, p.VELOCITY_CONTROL, force=0)  # force=0.5
            # enable joint torque sensing
            p.enableJointForceTorqueSensor(self.bot, i, 1)
            # increase max joint velocity (default = 100 rad/s)
            p.changeDynamics(self.bot, i, maxJointVelocity=800)  # max 3800 rpm

        self.X = np.zeros(13)  # initialize state

    def sim_run(self, u):
        q_ = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
        dq_ = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))

        q = q_.flatten()
        dq = dq_.flatten()
        qa = (q.T @ self.S).flatten()
        dqa = (dq_.T @ self.S).flatten()

        tau_s = self.spring_fn(q)

        tau = np.zeros(self.n_a)
        i = np.zeros(self.n_a)
        v = np.zeros(self.n_a)

        u *= -1
        tau[0], i[0], v[0] = self.actuator_q0.actuate(i=u[0], q_dot=dqa[0]) + tau_s[0]
        tau[1], i[1], v[1] = self.actuator_q2.actuate(i=u[1], q_dot=dqa[1]) + tau_s[1]
        tau[2], i[2], v[2] = self.actuator_rw1.actuate(i=u[2], q_dot=dqa[2])
        tau[3], i[3], v[3] = self.actuator_rw2.actuate(i=u[3], q_dot=dqa[3])
        tau[4], i[4], v[4] = self.actuator_rwz.actuate(i=u[4], q_dot=dqa[4])

        # tau[4] *= 0
        torque = self.S @ tau

        p.setJointMotorControlArray(self.bot, self.jointArray, p.TORQUE_CONTROL, forces=torque)

        Q_base_p = np.array(p.getBasePositionAndOrientation(self.bot)[1])
        Q_base = np.roll(Q_base_p, 1)  # pybullet gives quaternions in xyzw format instead of wxyz, shift values.
        velocities = p.getBaseVelocity(self.bot)
        self.X[0:3] = np.array(p.getBasePositionAndOrientation(self.bot)[0])
        self.X[3:7] = Q_base
        self.X[7:10] = Z(Q_inv(Q_base), velocities[0])  # linear vel world -> body frame
        self.X[10:] = Z(Q_inv(Q_base), velocities[1])  # angular vel world -> body frame

        f_sens, tau_sens = reaction(self.numJoints, self.bot)
        contact = np.array(p.getContactPoints(self.bot, self.plane, self.c_link), dtype=object)
        if np.shape(contact)[0] == 0:  # prevent empty list from being passed
            grf = np.zeros((3, 1))
            c = False
        else:
            grf_nrml_onB = np.array(contact[0, 7])
            grf_nrml = contact[0, 9]
            fric1 = contact[0, 10]
            fric1_dir = np.array(contact[0, 11])
            fric2 = contact[0, 12]
            fric2_dir = np.array(contact[0, 13])
            grf_z = grf_nrml * grf_nrml_onB
            fric_y = fric1 * fric1_dir
            fric_x = fric2 * fric2_dir
            grf = (grf_z + fric_y + fric_x).T
            c = True  # Detect contact with ground plane

        if useRealTime == 0:
            p.stepSimulation()

        return self.X, qa, dqa, c, tau, f_sens, tau_sens, i, v, grf

