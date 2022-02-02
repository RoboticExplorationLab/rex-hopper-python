import numpy as np


class Actuator:
    def __init__(self, i_max, gr_out, tau_stall, omega_max, **kwargs):
        """
        gr_out = gear ratio of output
        """
        self.i_max = i_max
        self.gr_out = gr_out
        self.tau_stall = tau_stall
        self.omega_max = omega_max  # 192 * 7 * (2 * np.pi / 60)  # rated speed, rpm to radians/s
        # self.kt_m = tau_stall * r / (v_max*gr)  # torque constant of the motor, Nm/amp. == v_max/omega_max
        self.kt_m = tau_stall / i_max  # 3.85 / 7
        # self.kt_m = self.v_max/omega_max
        self.v_max = omega_max * self.kt_m  # absolute maximum
        # self.omega_max = self.v_max / self.kt_m
        # r = (v_max ** 2) / (omega_max * tau_stall)
        self.r = self.kt_m * self.v_max / tau_stall

        # predicted final current and voltage of the motor
        self.i_actual = 0
        self.v_actual = 0

    def actuate(self, i, q_dot):
        """
        Motor Dynamics
        i = current, Amps
        q_dot = angular velocity of link (rad/s)
        omega = angular speed of motor (rad/s)
        """
        v_max = self.v_max
        gr_out = self.gr_out
        tau_stall = self.tau_stall
        kt_m = self.kt_m
        r = self.r
        omega = q_dot * gr_out  # convert link velocity to motor vel (thru gear ratio)
        tau_m = kt_m * i
        v = np.sign(i)*v_max
        tau_max_m = (- omega * (kt_m ** 2) + v * kt_m) / r  # max motor torque for given speed
        tau_min_m = (- omega * (kt_m ** 2) - v * kt_m) / r  # min motor torque for given speed
        if tau_max_m >= tau_min_m:
            tau_m = np.clip(tau_m, tau_min_m, tau_max_m)  # np.clip(tau_m, -abs(tau_max_m), abs(tau_max_m))
        else:
            tau_m = np.clip(tau_m, tau_max_m, tau_min_m)
        tau_m = np.clip(tau_m, -tau_stall, tau_stall)  # enforce max motor torque
        self.i_actual = tau_m / kt_m
        self.v_actual = kt_m * np.clip(omega, -self.omega_max, self.omega_max) + self.i_actual * r
        return tau_m * gr_out  # actuator output torque

    def actuate_sat(self, i, q_dot):
        """
        Uses simple inverse torque-speed relationship
        """
        gr_out = self.gr_out
        omega_max = self.omega_max  # motor rated speed, rad/s
        tau_stall = self.tau_stall
        omega = abs(q_dot * gr_out)
        # omega = abs(q_dot * gr_out)  # angular speed (NOT velocity) of motor in rad/s
        tau_max = (1-omega/omega_max) * tau_stall * gr_out
        return np.clip(i, -tau_max, tau_max)
