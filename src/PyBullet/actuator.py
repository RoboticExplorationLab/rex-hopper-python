import numpy as np

def actuate(v, q_dot, gr=7, gr_out=7, tau_stall=50, kt=3.85, r=0.38, l=0.45/1000, v_max=48):
    e = 0.97  # planetary gear efficiency assumed
    n_max = 190*(2*np.pi/60)*gr  # rated speed, rpm to radians and gear ratio
    # TODO: Why don't these work?
    # kt_m = kt/gr
    # kt_m = v_max/n_max  # torque constant of the motor, Nm/amp
    kt_m = tau_stall * r / (v_max*gr)
    v = np.clip(v, -v_max, v_max)
    omega = q_dot * gr_out  # angular velocity of motor in rad/s
    omega = np.clip(omega, -n_max, n_max)
    tau_m = - omega*(kt_m**2)/r + v*kt_m/r  # motor torque
    tau = tau_m * e * gr_out  # actuator output torque
    tau = np.clip(tau, -tau_stall, tau_stall)
    i = tau/kt

    return tau