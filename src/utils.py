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
    LQ[1:4, 1:4] = Q[0]*np.eye(3) + hat(Q[1:4])
    return LQ

def R(Q):
    RQ = np.zeros((4, 4))
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = - np.transpose(Q[1:4])
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0]*np.eye(3) - hat(Q[1:4])
    return RQ

H = np.zeros((4, 4))
H[1:4, 1:4] = np.eye(3)

T = np.zeros((4, 1))
T = np.fill_diagonal(T, [1.0, -1.0, -1.0, -1.0])

def Expq(phi):
    # The quaternion exponential map ϕ → q
    Q = np.zeros(4)
    theta = np.linalg.norm(phi)
    Q[0] = np.cos(theta / 2)
    Q[1:4] = 0.5 * phi * np.sinc(theta / (2 * np.pi))
    return Q

def Z(Q, p):
    # Rotate a position vector p by a quaternion Q
    return H.T * R(Q).T * L(Q) * H * p

def anglesolve(Q):
    # convert arbitrary quaternion to unsigned angle
    return 2 * np.atan2(np.linalg.norm(Q[1:4]), Q[0])

def angle_y(Q1, Q2):
    # signed angle about y axis of y-axis-constrained quaternions
    Q12 = L(Q1).T * Q2
    Q12 = Q12 / (np.linalg.norm(Q12))
    return 2 * np.arcsin(Q12[2])