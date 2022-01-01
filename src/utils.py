import numpy as np


def hat(w):
    # skew-symmetric
    return np.array([[0, - w[2], w[1]],
                     [w[2], 0, - w[0]],
                     [-w[1], w[0], 0]])


def L(Q):
    LQ = np.zeros((4, 4))
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = - np.transpose(Q[1:4])
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def R(Q):
    RQ = np.zeros((4, 4))
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = - np.transpose(Q[1:4])
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


H = np.zeros((4, 3))
H[1:4, 0:4] = np.eye(3)

T = np.zeros((4, 4))
np.fill_diagonal(T, [1.0, -1.0, -1.0, -1.0])


def Q_inv(Q):
    # Quaternion inverse
    Qinv = T @ Q
    return Qinv


def Expq(phi):
    # The quaternion exponential map ϕ → q
    Q = np.zeros(4)
    theta = np.linalg.norm(phi)
    Q[0] = np.cos(theta / 2)
    Q[1:4] = 0.5 * phi @ np.sinc(theta / (2 * np.pi))
    Q = Q / (np.linalg.norm(Q))  # re-normalize
    return Q


def Z(Q, p):
    # Rotate a position vector p by a quaternion Q
    return H.T @ R(Q).T @ L(Q) @ H @ p


def anglesolve(Q):
    # convert arbitrary quaternion to unsigned angle
    return 2 * np.arctan2(np.linalg.norm(Q[1:4]), Q[0])


def angle_y(Q1, Q2):
    # signed angle about y axis of y-axis-constrained quaternions
    Q12 = L(Q1).T @ Q2
    Q12 = Q12 / (np.linalg.norm(Q12))
    return 2 * np.arcsin(Q12[2])


def angle_z(Q1, Q2):
    # signed angle about z axis of z-axis-constrained quaternions
    Q12 = L(Q1).T @ Q2
    Q12 = Q12 / (np.linalg.norm(Q12))
    return 2 * np.arcsin(Q12[3])


def vec_to_quat(v2):
    # conversion of line vector to quaternion rotation b/t it and a datum vector v1
    v1 = np.array([1, 0, 0])  # datum vector, chosen as aligned with x-axis (front facing)
    Q = np.zeros(4)
    Q[0] = np.sqrt((np.linalg.norm(v1)**2)*(np.linalg.norm(v2)**2)) + np.dot(v1, v2)
    Q[1:4] = np.cross(v1, v2)
    Q = Q / np.linalg.norm(Q)
    return Q


def vec_to_quat2(v2):
    # conversion of line vector to quaternion rotation b/t it and a datum vector v1
    # alternative version
    v1 = np.array([0, 0, -1])  # datum vector, chosen as aligned with z-axis (representing leg direction)
    u1 = v1 / np.linalg.norm(v1)
    Q = np.zeros(4)
    if np.linalg.norm(v2) == 0 or np.isnan(v2).any():  # if input vector is zero or NaN, make Q default
        Q[0] = 1
    else:
        u2 = v2 / np.linalg.norm(v2)
        if np.array_equal(u1, -u2):
            Q[1:4] = np.linalg.norm(np.cross(v1, v2))
        else:
            u_half = (u1 + u2)/np.linalg.norm(u1 + u2)
            Q[0] = np.dot(u1, u_half)
            Q[1:4] = np.cross(u1, u_half)
            Q = Q / np.linalg.norm(Q)
    return Q_inv(Q)
