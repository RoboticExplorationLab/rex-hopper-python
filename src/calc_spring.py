import pickle
import dill
import sympy as sp
import numpy as np
from sympy.physics.vector import dynamicsymbols


def calculate(L, mass, I, coml):
    # --- Forward Kinematics --- #
    l0 = L[0]
    l1 = L[1]
    l2 = L[2]
    l3 = L[3]
    l4 = L[4]
    l5 = L[5]
    # lee = [l3 + l4, l5, 0]

    m0 = mass[0]
    m1 = mass[1]
    m2 = mass[2]
    m3 = mass[3]
    m = np.zeros((4, 4))
    np.fill_diagonal(m, [m0, m1, m2, m3])
    
    I0 = I[0]
    I1 = I[1]
    I2 = I[2]
    I3 = I[3]

    # CoM locations
    l_c0 = coml[0:3, 0]
    l_c1 = coml[0:3, 1]
    l_c2 = coml[0:3, 2]
    l_c3 = coml[0:3, 3]
    q0 = dynamicsymbols('q0')
    q1 = dynamicsymbols('q1')
    q2 = dynamicsymbols('q2')
    q3 = dynamicsymbols('q3')
    q0d = dynamicsymbols('q0d')
    q1d = dynamicsymbols('q1d')
    q2d = dynamicsymbols('q2d')
    q3d = dynamicsymbols('q3d')
    q0dd = sp.Symbol('q0dd')
    q1dd = sp.Symbol('q1dd')
    q2dd = sp.Symbol('q2dd')
    q3dd = sp.Symbol('q3dd')
    t = sp.Symbol('t')
    
    x0 = l_c0[0] * sp.cos(q0)
    y0 = l_c0[1]
    z0 = l_c0[2] * sp.sin(q0)
    
    x1 = l0 * sp.cos(q0) + l_c1[0] * sp.cos(q0 + q1)
    y1 = l_c1[1]
    z1 = l0 * sp.sin(q0) + l_c1[2] * sp.sin(q0 + q1)
    
    x2 = l_c2[0] * sp.cos(q2)
    y2 = l_c2[1]
    z2 = l_c2[2] * sp.sin(q2)
    
    x3 = l2 * sp.cos(q2) + l_c3[0] * sp.cos(q2 + q3)
    y3 = l_c3[1]
    z3 = l2 * sp.sin(q2) + l_c3[2] * sp.sin(q2 + q3)
    
    # Potential energy
    r0 = sp.Matrix([x0, y0, z0])
    r1 = sp.Matrix([x1, y1, z1])
    r2 = sp.Matrix([x2, y2, z2])
    r3 = sp.Matrix([x3, y3, z3])
    
    sp.var('gx gy gz')  # gravity vector
    g = sp.Matrix([gx, gy, gz])  # allows gravity vector to be updated
    U0 = m0 * g.dot(r0)
    U1 = m1 * g.dot(r1)
    U2 = m2 * g.dot(r2)
    U3 = m3 * g.dot(r3)
    U = U0 + U1 + U2 + U3
    
    # Kinetic energy
    x0d = sp.diff(x0, t)
    z0d = sp.diff(z0, t)
    x1d = sp.diff(x1, t)
    z1d = sp.diff(z1, t)
    x2d = sp.diff(x2, t)
    z2d = sp.diff(z2, t)
    x3d = sp.diff(x3, t)
    z3d = sp.diff(z3, t)
    
    v0_sq = x0d ** 2 + z0d ** 2
    v1_sq = x1d ** 2 + z1d ** 2
    v2_sq = x2d ** 2 + z2d ** 2
    v3_sq = x3d ** 2 + z3d ** 2
    T0 = 0.5 * m0 * v0_sq + 0.5 * I0 * q0d ** 2
    T1 = 0.5 * m1 * v1_sq + 0.5 * I1 * q1d ** 2
    T2 = 0.5 * m2 * v2_sq + 0.5 * I2 * q2d ** 2
    T3 = 0.5 * m3 * v3_sq + 0.5 * I3 * q3d ** 2
    T = T0 + T1 + T2 + T3
    
    # Le Lagrangian
    L = sp.trigsimp(T - U)
    L = L.subs(sp.Derivative(q0, t), q0d)  # substitute d/dt q2 with q2d
    L = L.subs(sp.Derivative(q1, t), q1d)  # substitute d/dt q1 with q1d
    L = L.subs(sp.Derivative(q2, t), q2d)  # substitute d/dt q2 with q2d
    L = L.subs(sp.Derivative(q3, t), q3d)  # substitute d/dt q2 with q2d
    
    # Euler-Lagrange Equation
    LE0 = sp.diff(sp.diff(L, q0d), t) - sp.diff(L, q0)
    LE1 = sp.diff(sp.diff(L, q1d), t) - sp.diff(L, q1)
    LE2 = sp.diff(sp.diff(L, q2d), t) - sp.diff(L, q2)
    LE3 = sp.diff(sp.diff(L, q3d), t) - sp.diff(L, q3)
    LE = sp.Matrix([LE0, LE1, LE2, LE3])
    
    # subs first derivative
    LE = LE.subs(sp.Derivative(q0, t), q0d)  # substitute d/dt q1 with q1d
    LE = LE.subs(sp.Derivative(q1, t), q1d)  # substitute d/dt q1 with q1d
    LE = LE.subs(sp.Derivative(q2, t), q2d)  # substitute d/dt q2 with q2d
    LE = LE.subs(sp.Derivative(q3, t), q3d)  # substitute d/dt q1 with q1d
    # subs second derivative
    LE = LE.subs(sp.Derivative(q0d, t), q0dd)  # substitute d/dt q1d with q1dd
    LE = LE.subs(sp.Derivative(q1d, t), q1dd)  # substitute d/dt q1d with q1dd
    LE = LE.subs(sp.Derivative(q2d, t), q2dd)  # substitute d/dt q2d with q2dd
    LE = LE.subs(sp.Derivative(q3d, t), q3dd)  # substitute d/dt q1d with q1dd
    LE = sp.expand(sp.simplify(LE))
    
    # Generalized mass matrix
    M = sp.zeros(4, 4)
    M[0, 0] = sp.collect(LE[0], q0dd).coeff(q0dd)
    M[0, 1] = sp.collect(LE[0], q1dd).coeff(q1dd)
    M[0, 2] = sp.collect(LE[0], q2dd).coeff(q2dd)
    M[0, 3] = sp.collect(LE[0], q3dd).coeff(q3dd)
    M[1, 0] = sp.collect(LE[1], q0dd).coeff(q0dd)
    M[1, 1] = sp.collect(LE[1], q1dd).coeff(q1dd)
    M[1, 2] = sp.collect(LE[1], q2dd).coeff(q2dd)
    M[1, 3] = sp.collect(LE[1], q3dd).coeff(q3dd)
    M[2, 0] = sp.collect(LE[2], q0dd).coeff(q0dd)
    M[2, 1] = sp.collect(LE[2], q1dd).coeff(q1dd)
    M[2, 2] = sp.collect(LE[2], q2dd).coeff(q2dd)
    M[2, 3] = sp.collect(LE[2], q3dd).coeff(q3dd)
    M[3, 0] = sp.collect(LE[3], q0dd).coeff(q0dd)
    M[3, 1] = sp.collect(LE[3], q1dd).coeff(q1dd)
    M[3, 2] = sp.collect(LE[3], q2dd).coeff(q2dd)
    M[3, 3] = sp.collect(LE[3], q3dd).coeff(q3dd)
    M_init = sp.lambdify([q0, q1, q2, q3], M)
    
    # Gravity Matrix
    G = LE
    G = G.subs(q0d, 0)
    G = G.subs(q1d, 0)  # must remove q derivative terms manually
    G = G.subs(q2d, 0)
    G = G.subs(q3d, 0)
    G = G.subs(q0dd, 0)
    G = G.subs(q1dd, 0)
    G = G.subs(q2dd, 0)
    G = G.subs(q3dd, 0)
    G_init = sp.lambdify([q0, q1, q2, q3, gx, gy, gz], G)
    
    # Coriolis Matrix
    # assume anything without qdd minus G is C
    C = LE
    C = C.subs(q0dd, 0)
    C = C.subs(q1dd, 0)
    C = C.subs(q2dd, 0)
    C = C.subs(q3dd, 0)
    C = C - G
    C_init = sp.lambdify([q0, q1, q2, q3, q0d, q1d, q2d, q3d], C)
    
    # --- Full Jacobian --- #
    rf = sp.Matrix([r0, r1, r2, r3])  # full kinematics, should be 12x1
    Jf = rf.jacobian([q0, q1, q2, q3])
    Jf_init = sp.lambdify([q0, q1, q2, q3], Jf)  # should be 12x4
    
    # compute del/delq(D(q)q_dot)q_dot of full jacobian
    q_dot = sp.Matrix([q0d, q1d, q2d, q3d])
    Jf_dqdot = Jf.multiply(q_dot)
    df = Jf_dqdot.jacobian([q0, q1, q2, q3]) * q_dot
    df_init = sp.lambdify([q0, q1, q2, q3, q0d, q1d, q2d, q3d], df)
    
    # --- Constraint --- #
    # constraint forward kinematics
    x1c = l0 * sp.cos(q0) + l1 * sp.cos(q0 + q1)
    z1c = l0 * sp.sin(q0) + l1 * sp.sin(q0 + q1)
    x2c = l2 * sp.cos(q2) + l3 * sp.cos(q2 + q3)
    z2c = l2 * sp.sin(q2) + l3 * sp.sin(q2 + q3)
    
    # compute constraint
    c = sp.zeros(2, 1)
    c[0] = x1c - x2c
    c[1] = z1c - z2c
    
    # constraint jacobian
    D = c.jacobian([q0, q1, q2, q3])
    D_init = sp.lambdify([q0, q1, q2, q3], D)
    
    # compute del/delq(D(q)q_dot)q_dot
    D_dqdot = D.multiply(q_dot)
    d = D_dqdot.jacobian([q0, q1, q2, q3]) * q_dot
    d_init = sp.lambdify([q0, q1, q2, q3, q0d, q1d, q2d, q3d], d)
    
    # compute cdot (first derivative of constraint function)
    cdot = sp.transpose(q_dot.T * D.T)
    cdot_init = sp.lambdify([q0, q1, q2, q3, q0d, q1d, q2d, q3d], cdot)
    
    # --- actuator forward kinematics --- #
    d = 0
    x0a = l0 * sp.cos(q0)
    z0a = l0 * sp.sin(q0)
    rho = sp.sqrt((x0a + d) ** 2 + z0a ** 2)
    x1a = l2 * sp.cos(q2)
    z1a = l2 * sp.sin(q2)
    h = sp.sqrt((x0a - x1a) ** 2 + (z0a - z1a) ** 2)
    mu = sp.acos((l3 ** 2 + h ** 2 - l1 ** 2) / (2 * l3 * h))
    eta = sp.acos((h ** 2 + l2 ** 2 - rho ** 2) / (2 * h * l2))
    alpha = sp.pi - (eta + mu) + q2
    xa = l2 * sp.cos(q2) + (l3 + l4) * sp.cos(alpha) - d + l5 * sp.cos(alpha - sp.pi / 2)
    ya = 0
    za = l2 * sp.sin(q2) + (l3 + l4) * sp.sin(alpha) + l5 * sp.cos(alpha - sp.pi / 2)
    fwd_kin = sp.Matrix([xa, ya, za])
    pos_init = sp.lambdify([q0, q2], fwd_kin)
    
    # compute end effector actuator jacobian
    Ja = fwd_kin.jacobian([q0, q2])
    Ja_init = sp.lambdify([q0, q2], Ja)
    
    # compute del/delq(Ja(q)q_dot)q_dot of ee actuator jacobian
    qa_dot = sp.Matrix([q0d, q2d])
    Ja_dqdot = Ja.multiply(qa_dot)
    da = Ja_dqdot.jacobian([q0, q2]) * qa_dot
    da_init = sp.lambdify([q0, q2, q0d, q2d], da)

    data = [M_init, G_init, C_init, Jf_init, df_init, D_init, d_init, cdot_init, pos_init, Ja_init, da_init]
    pik = "data.pickle"
    dill.settings['recurse'] = True
    with open(pik, 'wb') as f:
        dill.dump(data, f)

    return M_init, G_init, C_init, Jf_init, df_init, D_init, d_init, cdot_init, pos_init, Ja_init, da_init