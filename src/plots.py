"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use(['science', 'no-latex'])
plt.rcParams['lines.linewidth'] = 2
import matplotlib.ticker as plticker

plt.rcParams['font.size'] = 16


def thetaplot(total, thetahist, setphist, tauhist, dqhist):
    fig, axs = plt.subplots(3, 3, sharex='all', sharey='row')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), thetahist[:, 0] * 180 / np.pi, color='blue', label='Measurement')
    axs[0, 0].plot(range(total), setphist[:, 0] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_title('Theta 1')
    axs[0, 0].set_ylabel('Angle (deg)')

    axs[0, 1].plot(range(total), thetahist[:, 1] * 180 / np.pi, color='blue', label='Measurement')
    axs[0, 1].plot(range(total), setphist[:, 1] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 1].set_title('Theta 2')

    axs[0, 2].plot(range(total), thetahist[:, 2] * 180 / np.pi, color='blue', label='Measurement')
    axs[0, 2].plot(range(total), setphist[:, 2] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 2].set_title('Yaw')

    axs[1, 0].plot(range(total), tauhist[:, 2], color='orange')
    axs[1, 0].set_title('rw0')
    axs[1, 0].set_ylabel('Torque (Nm)')

    axs[1, 1].plot(range(total), tauhist[:, 3], color='orange')
    axs[1, 1].set_title('rw1')

    axs[1, 2].plot(range(total), tauhist[:, 4], color='orange')
    axs[1, 2].set_title('rwz')

    axs[2, 0].plot(range(total), dqhist[:, 2] * 60 / (2 * np.pi), color='g')
    axs[2, 0].set_title('rw0')
    axs[2, 0].set_ylabel('Angular Vel (RPM')

    axs[2, 1].plot(range(total), dqhist[:, 3] * 60 / (2 * np.pi), color='g')
    axs[2, 1].set_title('rw0')

    axs[2, 2].plot(range(total), dqhist[:, 4] * 60 / (2 * np.pi), color='g')
    axs[2, 2].set_title('rw0')

    plt.show()


def tauplot(model, total, n_a, tauhist):
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
        ax.set_title(model["aname"][k])

    plt.xlabel("Timesteps")
    plt.show()


def dqplot(model, total, n_a, dqhist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, dqhist[:, k] * 60 / (2 * np.pi))
        ax.set_ylabel('Angular Velocity, RPM')
        ax.set_title(model["aname"][k])
    plt.xlabel("Timesteps")
    plt.show()


def u_plot(total, u_hist, grfhist, p_hist, s_hist):
    fig, axs = plt.subplots(5, sharex='all')
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), p_hist[:, 2], color='blue')
    axs[0].set_title('base z position')
    axs[0].set_ylabel("z position (m)")

    axs[1].plot(range(total), grfhist[:, 0], color='b', label="Actual GRF")
    axs[1].plot(range(total), u_hist[:, 0], color='r', label="Force Ref")
    axs[1].set_title('X Ground Reaction Force')
    axs[1].set_ylabel("Force, N")

    axs[2].plot(range(total), grfhist[:, 1], color='b')
    axs[2].plot(range(total), u_hist[:, 1], color='r')
    axs[2].set_title('Y Ground Reaction Force')
    axs[2].set_ylabel("Force, N")

    axs[3].plot(range(total), grfhist[:, 2], color='b')
    axs[3].plot(range(total), u_hist[:, 2], color='r')
    axs[3].set_title('Z Ground Reaction Force')
    axs[3].set_ylabel("Force, N")

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center')

    axs[4].plot(range(total), s_hist[:, 0], color='b', lw='2', ls="--", label='Scheduled')
    axs[4].plot(range(total), s_hist[:, 1], color='r', lw='1', ls="-", label='Actual')
    axs[4].set_title('Scheduled Contact')
    axs[4].set_ylabel("True/False")
    axs[4].legend(loc="upper left")

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


def etotalplot(total, ahist, vhist, dt):
    ainhist = np.sum(ahist, axis=1)
    vmeanhist = np.average(vhist, axis=1)
    power_array = ahist @ vhist.T
    powerhist = np.diag(power_array)
    fig, axs = plt.subplots(3)
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), ainhist, color='blue', label='a_in')
    axs[0].set_title('Total Current Draw')
    axs[0].set_ylabel("Current (A)")

    print("Mean current draw is ", np.mean(ainhist), " A")
    print("Peak current draw is ", np.amax(ainhist), " A")

    axs[1].plot(range(total), vmeanhist, color='blue', label='v_mean')
    axs[1].set_title('Mean Voltage draw')
    axs[1].set_ylabel("Voltage (V)")

    axs[2].plot(range(total), powerhist, color='blue', label='power in')
    axs[2].set_title('Total Power')
    axs[2].set_ylabel("Power (W)")

    print("Mean power draw is ", np.mean(powerhist), " W")
    print("Peak power draw is ", np.amax(powerhist), " W")
    energy = np.trapz(powerhist, dx=dt)
    print("Total energy used is ", energy, " Joules, or ", energy / (48 * 3600), " Ah in ", np.shape(powerhist)[0] * dt,
          " s")
    plt.show()


def set_axes_equal(ax: plt.Axes):
    """
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def posplot_3d(p_hist, ref_traj, pf_ref):
    ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(*ref_traj[-1, 0:3], marker="x", s=200, color='orange', label='Target Position')
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], color='green', label='Reference Trajectory')
    ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='blue', label='Planned Footsteps')
    ax.scatter(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='CoM Position')
    ax.legend()
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    ax.set_box_aspect([1, 1, 1])  # make aspect ratio equal for all axes
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax)  # IMPORTANT - this is also required

    plt.show()


def animate_line(N, dataSet1, dataSet2, dataSet3, line, ref, pf, ax):
    line._offsets3d = (dataSet1[0:3, :N])
    ref._offsets3d = (dataSet2[0:3, :N])
    pf._offsets3d = (dataSet3[0:3, :N])
    ax.view_init(elev=10., azim=N)


def posplot_animate(p_hist, ref_traj, pf_ref):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, 2)
    ax.set_ylim3d(0, 2)
    ax.set_zlim3d(0, 2)

    ax.scatter(*p_hist[0, :], color='green', marker="x", s=200, label='Starting Position')
    ax.scatter(*ref_traj[-1, 0:3], marker="x", s=200, color='orange', label='Target Position')
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    N = len(p_hist)
    line = ax.scatter(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], lw=2, c='r', label='CoM Position')  # For line plot
    ref = ax.scatter(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], lw=2, c='g', label='Reference Trajectory')
    pf = ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='blue', label='Planned Footsteps')
    ax.legend()
    line_ani = animation.FuncAnimation(fig, animate_line, frames=N,
                                       fargs=(p_hist.T, ref_traj.T, pf_ref.T, line, ref, pf, ax),
                                       interval=2, blit=False)
    # line_ani.save('basic_animation.mp4', fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])

    plt.show()
