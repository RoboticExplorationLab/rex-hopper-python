import numpy as np


class Actuator:
    def __init__(self, dt, model, **kwargs):
        """
        gr = gear ratio of output
        """
        self.v_max = model["v_max"]  # omega_max * self.kt  # absolute maximum
        self.gr = model["gr"]
        basis = "ele"

        if basis == "ele":
            self.i_max = model["i_max"]
            self.r = model["r"]
            self.kt = model["kt"]
            self.tau_max = self.i_max * self.kt  # absolute max backdriving motor torque
            self.omega_max = self.v_max / self.kt

        elif basis == "tau":
            self.i_max = model["i_max"]
            self.tau_max = model["tau_max"]
            self.omega_max = model["omega_max"]
            self.kt = self.tau_max / self.i_max  # self.v_max/self.omega_max
            self.r = self.kt * self.v_max / self.tau_max  # (v_max ** 2) / (omega_max * tau_max)
            # self.tau_max = self.i_max * self.kt  # absolute max backdriving motor torque

        elif basis == "vel":
            self.i_max = model["i_max"]
            tau_max = model["tau_max"]
            self.omega_max = model["omega_max"]
            self.r = (self.v_max ** 2) / (self.omega_max * tau_max)
            self.kt = self.v_max/self.omega_max
            self.tau_max = self.i_max * self.kt  # absolute max backdriving motor torque

        # predicted final current and voltage of the motor
        self.i_actual = np.zeros(2)
        self.v_actual = np.zeros(2)

        # smoothing bandwidth
        self.i_smoothed = 0
        dt = dt
        tau = 1/160  # inverse of Odrive torque bandwidth
        # G(s) = tau/(s+tau)  http://techteach.no/simview/lowpass_filter/doc/filter_algorithm.pdf
        self.alpha = dt / (dt + tau)  # DT version of low pass filter

    def actuate(self, i, q_dot):
        """
        Motor Dynamics
        i = current, Amps
        q_dot = angular velocity of link (rad/s)
        omega = angular speed of motor (rad/s)
        """
        v_max = self.v_max
        gr = self.gr
        # tau_max = self.tau_max
        tau_max = self.tau_max
        kt = self.kt
        r = self.r
        alpha = self.alpha
        omega = q_dot * gr  # convert link velocity to motor vel (thru gear ratio)

        self.i_smoothed = (1-alpha)*self.i_smoothed + alpha*i
        i = self.i_smoothed

        tau_m = kt * i
        v = np.sign(i)*v_max
        tau_ul = (- omega * (kt ** 2) + v * kt) / r  # max motor torque for given speed
        tau_ll = (- omega * (kt ** 2) - v * kt) / r  # min motor torque for given speed
        if tau_ul >= tau_ll:
            tau_m = np.clip(tau_m, tau_ll, tau_ul)  # np.clip(tau_m, -abs(tau_ul), abs(tau_ul))
        else:
            tau_m = np.clip(tau_m, tau_ul, tau_ll)

        tau_m = np.clip(tau_m, -tau_max, tau_max)  # enforce max motor torque

        self.i_actual = abs(tau_m / kt)
        v_actual_in = abs(self.i_actual * r + kt * np.clip(omega, -self.omega_max, self.omega_max))
        v_actual_backemf = abs(kt * np.clip(omega, -self.omega_max, self.omega_max))
        self.v_actual = np.array([v_actual_in, v_actual_backemf]).flatten()

        return tau_m * gr  # actuator output torque

    def actuate_sat(self, i, q_dot):
        """
        Uses simple inverse torque-speed relationship
        """
        gr = self.gr
        omega_max = self.omega_max  # motor rated speed, rad/s
        tau_max = self.tau_max
        omega = abs(q_dot * gr)
        # omega = abs(q_dot * gr)  # angular speed (NOT velocity) of motor in rad/s
        tau_max = (1-omega/omega_max) * tau_max * gr
        return np.clip(i, -tau_max, tau_max)
