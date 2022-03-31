"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt


def thetaplot(total, thetahist, setphist):

    fig, axs = plt.subplots(1, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), thetahist[:, 0]*180/np.pi, color='blue', label='Measurement')
    axs[0].plot(range(total), setphist[:, 0] * 180 / np.pi, color='red', label='Setpoint')
    axs[0].legend(loc="lower right")
    axs[0].set_title('Theta_1')
    axs[0].set_ylabel('Theta_1 (deg)')

    axs[1].plot(range(total), thetahist[:, 1]*180/np.pi, color='blue', label='Measurement')
    axs[1].plot(range(total), setphist[:, 1] * 180 / np.pi, color='red', label='Setpoint')
    axs[1].legend(loc="lower right")
    axs[1].set_title('Theta_2')
    axs[1].set_ylabel('Theta_2 (deg)')

    axs[2].plot(range(total), thetahist[:, 2]*180/np.pi, color='blue', label='Measurement')
    axs[2].plot(range(total), setphist[:, 2] * 180 / np.pi, color='red', label='Setpoint')
    axs[2].legend(loc="lower right")
    axs[2].set_title('Yaw')
    axs[2].set_ylabel('Yaw (deg)')
    plt.show()


def tauplot(total, n_a, tauhist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, tauhist[:, k])  
        ax.set_ylabel('Torque, Nm')
    plt.xlabel("Timesteps")
    plt.show()


def dqplot(total, n_a, dqhist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, dqhist[:, k]*60/(2*np.pi))  
        ax.set_ylabel('Angular Velocity, RPM')
    plt.xlabel("Timesteps")
    plt.show()


def fplot(total, phist, fhist, fthist):

    fig, axs = plt.subplots(4, sharex='all')
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), phist[:, 2], color='blue')
    axs[0].set_title('base z position')
    axs[0].set_ylabel("z position (m)")

    axs[1].plot(range(total), fhist[:, 0], color='blue')
    axs[1].set_title('Magnitude of X Reaction Force on joint1')
    axs[1].set_ylabel("Reaction Force Fx, N")

    axs[2].plot(range(total), fhist[:, 2], color='blue')
    axs[2].set_title('Magnitude of Z Reaction Force on joint1')  # .set_title('angular velocity q1_dot')
    axs[2].set_ylabel("Reaction Force Fz, N")  # .set_ylabel("angular velocity, rpm")

    axs[3].plot(range(total), fthist, color='blue')
    axs[3].set_title('Flight Time')  # .set_title('angular velocity q1_dot')
    axs[3].set_ylabel("Time, s")  # .set_ylabel("angular velocity, rpm")

    plt.show()


def currentplot(total, n_a, ahist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, ahist[:, k])
        ax.set_ylabel("current (A)")
    plt.xlabel("Timesteps")
    plt.show()


def voltageplot(total, n_a, vhist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, vhist[:, k])
        ax.set_ylabel("voltage (V)")
    plt.xlabel("Timesteps")
    plt.show()


def electrtotalplot(total, ahist, vhist, dt):

    ainhist = np.sum(ahist, axis=1)
    vmeanhist = np.average(vhist, axis=1)
    power_array = ahist @ vhist.T
    powerhist = np.diag(power_array)
    fig, axs = plt.subplots(3)
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), ainhist, color='blue', label='a_in')
    axs[0].set_title('total current draw')
    axs[0].set_ylabel("current (A)")

    print("Mean current draw is ", np.mean(ainhist), " A")
    print("Peak current draw is ", np.amax(ainhist), " A")

    axs[1].plot(range(total), vmeanhist, color='blue', label='v_mean')
    axs[1].set_title('mean voltage draw')
    axs[1].set_ylabel("voltage (V)")

    axs[2].plot(range(total), powerhist, color='blue', label='power in')
    axs[2].set_title('total power')
    axs[2].set_ylabel("power (W)")

    print("Mean power draw is ", np.mean(powerhist), " W")
    print("Peak power draw is ", np.amax(powerhist), " W")
    energy = np.trapz(powerhist, dx=dt)
    print("Total energy used is ", energy, " Joules, or ", energy/(48*3600), " Ah in ", np.shape(powerhist)[0]*dt, " s")
    plt.show()


def posplot(p_ref, phist, x_des_hist):

    plt.plot(phist[:, 0], phist[:, 1], color='blue', label='body position')
    plt.title('Body XY Position')
    plt.ylabel("y (m)")
    plt.xlabel("x (m)")
    plt.scatter(x_des_hist[:, 0], x_des_hist[:, 1], color='red', marker="o", label='footstep setpoints')
    plt.scatter(0, 0, color='green', marker="x", s=100, label='starting position')
    plt.scatter(p_ref[0], p_ref[1], color='orange', marker="x", s=100, label='position setpoint')
    plt.legend(loc="upper left")

    plt.show()


def posplot_3d(p_ref, phist, x_des_hist):
    ax = plt.axes(projection='3d')
    ax.plot(phist[:, 0], phist[:, 1], phist[:, 2], color='red', label='body position')
    ax.set_title('Body XYZ Position')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.scatter(0, 0, 0, color='green', label='starting position')
    ax.scatter(p_ref[0], p_ref[1], p_ref[2], color='orange', label='position setpoint')
    ax.scatter(x_des_hist[:, 0], x_des_hist[:, 1], x_des_hist[:, 2], color='blue', label='foot position')
    ax.legend()

    plt.show()
