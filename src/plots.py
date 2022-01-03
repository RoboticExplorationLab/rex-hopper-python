"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt


def rwplot(total, hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, setphist1, setphist2, setphist3):

    fig, axs = plt.subplots(3, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), hist1*180/np.pi, color='blue', label='Measurement')
    axs[0, 0].plot(range(total), setphist1 * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].set_title('Theta_1')
    axs[0, 0].set_ylabel('Theta_1 (deg)')

    axs[0, 1].plot(range(total), hist2*180/np.pi, color='blue', label='Measurement')
    axs[0, 1].plot(range(total), setphist2 * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].set_title('Theta_2')
    axs[0, 1].set_ylabel('Theta_2 (deg)')

    axs[0, 2].plot(range(total), hist3*180/np.pi, color='blue', label='Measurement')
    axs[0, 2].plot(range(total), setphist3 * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 2].legend(loc="lower right")
    axs[0, 2].set_title('Yaw')
    axs[0, 2].set_ylabel('Yaw (deg)')

    axs[1, 0].plot(range(total), hist4, color='blue')
    axs[1, 0].set_title('RW1 Torque')
    axs[1, 0].set_ylabel('Torque, Nm')

    axs[1, 1].plot(range(total), hist5, color='blue')
    axs[1, 1].set_title('RW1 Torque')
    axs[1, 1].set_ylabel('Torque, Nm')

    axs[1, 2].plot(range(total), hist6, color='blue')
    axs[1, 2].set_title('RWZ Torque')
    axs[1, 2].set_ylabel('Torque, Nm')

    axs[2, 0].plot(range(total), hist7*60/(2*np.pi), color='blue')
    axs[2, 0].set_title('RW1 Angular Velocity')
    axs[2, 0].set_ylabel('Angular Velocity, RPM')

    axs[2, 1].plot(range(total), hist8*60/(2*np.pi), color='blue')
    axs[2, 1].set_title('RW2 Angular Velocity')
    axs[2, 1].set_ylabel('Angular Velocity, RPM')

    axs[2, 2].plot(range(total), hist9*60/(2*np.pi), color='blue')
    axs[2, 2].set_title('RWZ Angular Velocity')
    axs[2, 2].set_ylabel('Angular Velocity, RPM')

    plt.show()


def thetaplot(total, hist1, hist2, hist3, hist4, hist5, hist6):

    fig, axs = plt.subplots(2, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), hist1*180/np.pi, color='blue')
    axs[0, 0].set_title('Theta_x')
    axs[0, 0].set_ylabel('Theta_x (deg)')

    axs[0, 1].plot(range(total), hist2*180/np.pi, color='blue')
    axs[0, 1].set_title('Theta_y')
    axs[0, 1].set_ylabel('Theta_y (deg)')

    axs[0, 2].plot(range(total), hist3*180/np.pi, color='blue')
    axs[0, 2].set_title('Theta_z')
    axs[0, 2].set_ylabel('Theta_z (deg)')

    axs[1, 0].plot(range(total), hist4*60/(2*np.pi), color='blue')
    axs[1, 0].set_title('RW1 Angular Velocity')
    axs[1, 0].set_ylabel('Angular Velocity, RPM')

    axs[1, 1].plot(range(total), hist5*60/(2*np.pi), color='blue')
    axs[1, 1].set_title('RW2 Angular Velocity')
    axs[1, 1].set_ylabel('Angular Velocity, RPM')

    axs[1, 2].plot(range(total), hist6*60/(2*np.pi), color='blue')
    axs[1, 2].set_title('RWZ Angular Velocity')
    axs[1, 2].set_ylabel('Angular Velocity, RPM')

    plt.show()


def tauplot(total, hist1, hist2, hist3, hist4, hist5, hist6):

    fig, axs = plt.subplots(2, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), hist1, color='blue')
    axs[0, 0].set_title('q0 torque')
    axs[0, 0].set_ylabel("q0 torque (Nm)")

    axs[0, 1].plot(range(total), hist2, color='blue')
    axs[0, 1].set_title('q1 torque')
    axs[0, 1].set_ylabel("q1 torque (Nm)")

    axs[0, 2].plot(range(total), hist3, color='blue')
    axs[0, 2].set_title('base z position')
    axs[0, 2].set_ylabel("z position (m)")

    axs[1, 0].plot(range(total), hist4, color='blue')
    axs[1, 0].set_title('Magnitude of X Reaction Force on joint1')
    axs[1, 0].set_ylabel("Reaction Force Fx, N")

    axs[1, 1].plot(range(total), hist5, color='blue')
    axs[1, 1].set_title('Magnitude of Z Reaction Force on joint1')  # .set_title('angular velocity q1_dot')
    axs[1, 1].set_ylabel("Reaction Force Fz, N")  # .set_ylabel("angular velocity, rpm")

    axs[1, 2].plot(range(total), hist6, color='blue')
    axs[1, 2].set_title('Flight Time')
    axs[1, 2].set_ylabel("Flight Time, s")

    plt.show()


def posplot(p_ref, phist, xfhist):

    plt.plot(phist[:, 0], phist[:, 1], color='blue', label='body position')
    plt.title('Body XY Position')
    plt.ylabel("y (m)")
    plt.xlabel("x (m)")
    plt.scatter(xfhist[:, 0], xfhist[:, 1], color='red', marker="o", label='footstep setpoints')
    plt.scatter(0, 0, color='green', marker="x", s=100, label='starting position')
    plt.scatter(p_ref[0], p_ref[1], color='orange', marker="x", s=100, label='position setpoint')
    plt.legend(loc="upper left")

    plt.show()
