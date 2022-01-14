"""
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np
import transforms3d
import pybullet as p
import pybullet_data
import os

import actuator

useRealTime = 0  # Do NOT change to real time


def spring(q, l):
    """
    adds linear extension spring b/t joints 1 and 3 of parallel mechanism
    approximated by applying torques to joints 0 and 2
    """
    init_q = [-30 * np.pi / 180, -150 * np.pi / 180]
    if q is None:
        q0 = init_q[0]
        q2 = init_q[2]
    else:
        q0 = q[0] + init_q[0]
        q2 = q[2] + init_q[1]
    k = 1500  # spring constant, N/m
    L0 = l[0]  # .15
    L2 = l[2]  # .3
    gamma = abs(q2 - q0)
    rmin = 0.204*0.8
    r = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(gamma))  # length of spring
    print("r = ", r, " and rmin = ", rmin)
    if r < rmin:
        print("error: incorrect spring params, r = ", r, " and rmin = ", rmin)
    T = k * (r - rmin)  # spring tension force
    alpha = np.arccos((-L0 ** 2 + L2 ** 2 + r ** 2) / (2 * L2 * r))
    beta = np.arccos((-L2 ** 2 + L0 ** 2 + r ** 2) / (2 * L0 * r))
    tau_s0 = -T * np.sin(beta) * L0
    tau_s1 = T * np.sin(alpha) * L2
    tau_s = np.array([tau_s0, tau_s1])
    return tau_s


def reaction_force(numJoints, bot):
    # returns joint reaction force
    reaction = np.array([j[2] for j in p.getJointStates(bot, range(numJoints))])  # j[2]=jointReactionForces
    # 4x6 array [Fx, Fy, Fz, Mx, My, Mz]
    f = reaction[:, 0:3]  # selected all joints Fz
    # 4x3 array [Fx, Fy, Fz]
    # f = np.linalg.norm(reaction[:, 0:3], axis=1)  # selected all joints Fz
    return f


class Sim:

    def __init__(self, model, dt=1e-3, fixed=False, spring=False, record=False, scale=1, gravoff=False, direct=False):
        self.dt = dt
        self.omega_xyz = None
        self.omega = None
        self.v = None
        self.record_rt = record  # record video in real time
        self.base_pos = None
        self.spring = spring
        self.L = model["linklengths"]
        self.dir_s = model["springpolarity"]

        omega_max_x10 = 190 * 7 * (2 * np.pi / 60)
        omega_max_rw = 190 * 7 * (2 * np.pi / 60)
        self.actuator_x10 = actuator.Actuator(v_max=48, gr_out=7, tau_stall=50 / 7, omega_max=omega_max_x10)
        self.actuator_rw = actuator.Actuator(v_max=48, gr_out=7, tau_stall=50 / 7, omega_max=omega_max_rw)

        if gravoff == True:
            GRAVITY = 0
        else:
            GRAVITY = -9.807

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
        self.bot = p.loadURDF(os.path.join(path_parent, model_path), [0, 0, 0.5*scale],  # 0.31
                         robotStartOrientation, useFixedBase=fixed, globalScaling=scale,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)

        self.jointArray = range(p.getNumJoints(self.bot))
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.dt)
        self.numJoints = p.getNumJoints(self.bot)
        p.setRealTimeSimulation(useRealTime)

        self.c_link = 1
        self.model = model["model"]

        if self.model != 'design_rw':
            vert = p.createConstraint(self.bot, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 0])

        if self.model == 'design_rw':
            # p.createConstraint(self.bot, 3, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [-0.135, 0, 0], [0, 0, 0])
            jconn_1 = [x * scale for x in [0.135, 0, 0]]
            jconn_2 = [x * scale for x in [-0.0014381, 0, 0.01485326948]]
            linkjoint = p.createConstraint(self.bot, 1, self.bot, 3, p.JOINT_POINT2POINT, [0, 0, 0], jconn_1, jconn_2)
            p.changeConstraint(linkjoint, maxForce=1000)
            self.c_link = 3

        elif self.model == 'design':
            jconn_1 = [x*scale for x in [0.15, 0, 0]]
            jconn_2 = [x*scale for x in [-0.01317691945, 0, 0.0153328498]]
            linkjoint = p.createConstraint(self.bot, 1, self.bot, 3,
                                     p.JOINT_POINT2POINT, [0, 0, 0], jconn_1, jconn_2)
            p.changeConstraint(linkjoint, maxForce=1000)
            self.c_link = 3

        elif self.model == 'parallel':
            linkjoint = p.createConstraint(self.bot, 1, self.bot, 3,
                                     p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [.15, 0, 0])
            p.changeConstraint(linkjoint, maxForce=1000)

        elif self.model == 'belt':
            # vert = p.createConstraint(self.bot, -1, -1, 1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0.3, 0, 0])
            belt = p.createConstraint(self.bot, 0, self.bot, 1,
                                     p.JOINT_GEAR, [0, 1, 0], [0, 0, 0], [0, 0, 0])
            p.changeConstraint(belt, gearRatio=0.5, gearAuxLink=-1, maxForce=1000)

        # increase friction of toe to ideal
        # p.changeDynamics(self.bot, self.c_link, lateralFriction=2, contactStiffness=100000, contactDamping=10000)
        p.changeDynamics(self.bot, self.c_link, lateralFriction=3)  # , restitution=0.01)

        # Record Video in real time
        if self.record_rt is True:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        # Disable the default velocity/position motor:
        for i in range(self.numJoints):
            p.setJointMotorControl2(self.bot, i, p.VELOCITY_CONTROL, force=0)  # force=0.5
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(self.bot, i, 1)  # enable joint torque sensing

    def sim_run(self, u, u_rw):
        q_all = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
        q_dot_all = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))

        if self.spring:
            tau_s = spring(q_all, self.L) * self.dir_s
        else:
            tau_s = np.zeros(2)

        self.p = np.array(p.getBasePositionAndOrientation(self.bot)[0])
        Q_base_p = np.array(p.getBasePositionAndOrientation(self.bot)[1])
        # pybullet gives quaternions in xyzw format instead of wxyz, so you need to shift values
        Q_base = np.roll(Q_base_p, 1)  # move last element to first place
        q = np.zeros(self.numJoints)
        q_dot = np.zeros(self.numJoints)
        qrw = np.zeros(3)
        qrw_dot = np.zeros(3)
        torque = np.zeros(self.numJoints)
        command = np.zeros(self.numJoints)

        if self.model == "design_rw":
            command[0] = -u[0]  # readjust to match motor polarity
            command[2] = -u[1]  # readjust to match motor polarity
            q = q_all[0:4]
            q_dot = q_dot_all[0:4]
            qrw = q_all[4:]
            qrw_dot = q_dot_all[4:]
            torque[0] = self.actuator_x10.actuate(i=command[0], q_dot=q_dot[0]) + tau_s[0]
            torque[2] = self.actuator_x10.actuate(i=command[2], q_dot=q_dot[2]) + tau_s[1]
            torque[4] = self.actuator_rw.actuate(i=u_rw[0], q_dot=qrw_dot[0])
            torque[5] = self.actuator_rw.actuate(i=u_rw[1], q_dot=qrw_dot[1])
            torque[6] = self.actuator_rw.actuate(i=u_rw[2], q_dot=qrw_dot[2])

        if self.model == "design":
            command[0] = -u[0]  # readjust to match motor polarity
            command[2] = -u[1]  # readjust to match motor polarity
            q = q_all
            q_dot = q_dot_all
            torque[0] = self.actuator_x10.actuate(i=command[0], q_dot=q_dot[0]) + tau_s[0]
            torque[2] = self.actuator_x10.actuate(i=command[2], q_dot=q_dot[2]) + tau_s[1]

        elif self.model == "serial":
            command = -u
            q = q_all
            q_dot = q_dot_all
            torque[0] = self.actuator_x10.actuate(i=command[0], q_dot=q_dot[0])
            torque[1] = self.actuator_x10.actuate(i=command[1], q_dot=q_dot[1])

        elif self.model == "parallel":
            command[0] = -u[1]  # readjust to match motor polarity
            command[2] = -u[0]  # readjust to match motor polarity
            q[0] = q_all[2]
            q[2] = q_all[0]
            q_dot[0] = q_dot_all[2]
            q_dot[2] = q_dot_all[0]  # modified from [1] to [2] 11-5-21
            torque[0] = self.actuator_x10.actuate(i=command[0], q_dot=q_dot[0]) + tau_s[0]
            torque[2] = self.actuator_x10.actuate(i=command[2], q_dot=q_dot[2]) + tau_s[1]

        elif self.model == "belt":
            command[0] = -u[0]  # only 1 DoF actuated
            q = q_all[0]  # only 1 DoF actuated, remove extra.
            q_dot[0] = q_dot_all[0]
            torque[0] = self.actuator_x10.actuate(i=command[0], q_dot=q_dot[0])
            # torque[0] = actuator.actuate(i=command[0], q_dot=q_dot[0], gr_out=21)

        p.setJointMotorControlArray(self.bot, self.jointArray, p.TORQUE_CONTROL, forces=torque)
        velocities = p.getBaseVelocity(self.bot)
        self.v = velocities[0]  # base linear velocity in global Cartesian coordinates
        self.omega_xyz = velocities[1]  # base angular velocity in XYZ
        self.base_pos = p.getBasePositionAndOrientation(self.bot)
        f = reaction_force(self.numJoints, self.bot)
        # Detect contact with ground plane
        c = bool(len([c[8] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]))

        if useRealTime == 0:
            p.stepSimulation()

        return q, q_dot, qrw, qrw_dot, Q_base, c, torque, f
