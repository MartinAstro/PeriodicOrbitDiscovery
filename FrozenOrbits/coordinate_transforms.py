import numpy as np
import tensorflow as tf


def _convert_angle_positive(angle):
    zero = tf.constant(0, dtype=angle.dtype)
    pi = tf.constant(np.pi, dtype=angle.dtype)
    angle = tf.cond(angle < zero, lambda: angle + 2 * pi, lambda: angle)
    return angle


def make_angle_positive(angle):
    angle = tf.reshape(angle, [-1, 1])
    angle = tf.map_fn(fn=lambda angle: _convert_angle_positive(angle), elems=angle)
    angle = tf.reshape(angle, [-1, 1])
    return angle


def _convert_angle_cond_1(angle):
    pi = tf.constant(np.pi, dtype=angle.dtype)
    angle = tf.cond(angle > pi, lambda: angle - 2 * pi, lambda: angle)
    return angle


def _convert_angle_cond_2(angle):
    pi = tf.constant(np.pi, dtype=angle.dtype)
    angle = tf.cond(angle < -pi, lambda: angle + 2 * pi, lambda: angle)
    return angle


def convert_angle(angle):
    angle = tf.reshape(angle, [-1, 1])
    angle = tf.map_fn(fn=lambda angle: _convert_angle_cond_1(angle), elems=angle)
    angle = tf.map_fn(fn=lambda angle: _convert_angle_cond_2(angle), elems=angle)
    angle = tf.reshape(angle, [-1, 1])
    return angle


def sph2cart_tf(r_vec):
    r = r_vec[:, 0]  # [0, inf]
    pi = tf.constant(np.pi, dtype=r.dtype)
    theta = r_vec[:, 1] * pi / 180.0
    phi = r_vec[:, 2] * pi / 180.0

    x = r * tf.math.sin(phi) * tf.math.cos(theta)
    y = r * tf.math.sin(phi) * tf.math.sin(theta)
    z = r * tf.math.cos(phi)

    return tf.stack([x, y, z], 1)


def tf_dot(a, b):
    result = tf.reduce_sum(a * b, axis=1, keepdims=True)
    return result


def fFunc(M, E, e):  # Residual Function
    val = M - (E - e * tf.sin(E))
    return val


def fFuncH(M, H, e):  # Residual Function
    val = M - (e * tf.sinh(H) - H)
    return val


def _e_squared_m1_or_1m_e_squared_cond(e):
    e_squared_m1_or_1m_e_squared = tf.cond(
        e > 1.0,
        lambda: (tf.square(e) - 1.0),
        lambda: (1.0 - tf.square(e)),
    )
    return e_squared_m1_or_1m_e_squared


def e_squared_m1_or_1m_e_squared(e):
    e_squared_m1_or_1m_e_squared = tf.map_fn(
        fn=lambda e: _e_squared_m1_or_1m_e_squared_cond(e),
        elems=e,
    )
    return e_squared_m1_or_1m_e_squared


def _e_vec_normalization_cond(e_ijk):
    unit_vec = tf.constant([1.0, 0.0, 0.0], dtype=tf.float64)
    e_mag_lim = tf.constant([1e-14], dtype=tf.float64)
    e_mag = tf.linalg.norm(e_ijk)

    undefined_e_vec = tf.reduce_all(tf.math.is_nan(e_ijk), axis=0, keepdims=True)
    excessively_small_e_vec = tf.less(e_mag, e_mag_lim)
    e_vec_normalized = tf.cond(
        undefined_e_vec or excessively_small_e_vec,
        lambda: unit_vec,
        lambda: e_ijk / e_mag,
    )
    return e_vec_normalized


def e_vec_normalization(eVecList):
    e_vec_normalized = tf.map_fn(
        fn=lambda e_ijk: _e_vec_normalization_cond(e_ijk),
        elems=eVecList,
    )
    return e_vec_normalized


def _semi_multiplier_cond(e):
    semi_multiplier = tf.cond(e > 1.0, lambda: -1.0, lambda: 1.0)
    return semi_multiplier


# def semi_multiplier(e):
#     semi_multiplier = tf.map_fn(fn=lambda e: _semi_multiplier_cond(e), elems=e)
#     return semi_multiplier


def _semilatus_rectum(OE):
    a = OE[0]
    e = OE[1]
    e_squared_m1_or_1m_e_squared = _e_squared_m1_or_1m_e_squared_cond(e)
    # semi_multiplier = _semi_multiplier_cond(e)
    # p = semi_multiplier*a*e_squared_m1_or_1m_e_squared
    p = a * e_squared_m1_or_1m_e_squared
    return p


def semilatus_rectum(OE):
    p = tf.map_fn(fn=lambda OE: _semilatus_rectum(OE), elems=OE)
    return p


def _conditional_I(i):
    pi = tf.constant(np.pi, dtype=tf.float64)
    I = tf.cond(
        i > pi / 2.0,
        lambda: tf.constant(-1.0, dtype=tf.float64),
        lambda: tf.constant(1.0, dtype=tf.float64),
    )
    return I


def get_I(i):
    i = tf.map_fn(fn=lambda i: _conditional_I(i), elems=i)
    i = tf.reshape(i, [-1])
    return i


def _computeM_elliptic(f, e):
    E = 2.0 * tf.atan2(tf.sqrt((1.0 - e) / (1.0 + e)) * tf.tan(f / 2.0), 1.0)
    M = E - e * tf.sin(E)
    return M


def _computeM_hyperbolic(f, e):
    term1 = tf.sqrt((e - 1.0) / (e + 1.0)) * tf.tan(f / 2.0)
    H = 2.0 * tf.atanh(term1)
    M = e * tf.sinh(H) - H
    return M


def _conditional_mean_anomaly_map(elems):
    f = elems[0]
    e = elems[1]
    M = tf.cond(
        e > 1,
        lambda: _computeM_hyperbolic(f, e),
        lambda: _computeM_elliptic(f, e),
    )
    return M


def computeMeanAnomaly(f, e):
    f = tf.reshape(f, [-1, 1])
    e = tf.reshape(e, [-1, 1])
    elems = tf.stack((f, e), 1)
    M = tf.map_fn(fn=lambda elems: _conditional_mean_anomaly_map(elems), elems=(elems))
    M = tf.reshape(M, [-1])
    return M


def _computeEccentricAnomaly(M, e):
    E = tf.identity(M)
    for _ in range(10):
        f0 = fFunc(M, E, e)
        dfdE = -1.0 * (1.0 - e * tf.cos(E))
        deltaE = -f0 / dfdE
        E = E + deltaE
    return E


def _computeHyperbolicAnomaly(M, e):
    H = tf.identity(M)
    for _ in range(10):
        f0 = fFuncH(M, H, e)
        dfdH = 1.0 - e * tf.cosh(H)
        H = H - f0 / dfdH
    return H


def _computeTrueAnomaly_elliptic(OE):
    e = OE[1]
    M = OE[5]
    E = _computeEccentricAnomaly(M, e)
    f_New = 2.0 * tf.math.atan2(tf.sqrt((1.0 + e) / (1.0 - e)) * tf.tan(E / 2.0), 1.0)
    return f_New


def _computeTrueAnomaly_hyperbolic(OE):
    e = OE[1]
    M = OE[5]

    H = _computeHyperbolicAnomaly(M, e)
    f_New = 2.0 * tf.math.atan2(tf.sqrt((e + 1.0) / (e - 1.0)) * tf.tanh(H / 2.0), 1.0)
    return f_New


def _conditional_true_anomaly_map(OE):
    e = OE[1]
    f_New = tf.cond(
        tf.greater(e, 1),
        lambda: _computeTrueAnomaly_hyperbolic(OE),
        lambda: _computeTrueAnomaly_elliptic(OE),
    )
    return f_New


def computeTrueAnomaly(OE):
    f_New = tf.map_fn(fn=lambda OE: _conditional_true_anomaly_map(OE), elems=OE)
    # f_New = convert_angle(f_New)
    return f_New


# Delaunay
def oe2delaunay_tf(OE, mu):
    """Convert traditional elements to delaunay elements

    Args:
        trad_OE (tf.Tensor or np.array): [N x 6] (a, e, i, omega, Omega, M)

    Returns:
        delaunay_OE: [N x 6] (l, g, h, L, G, H)
    """
    a, e, i, omega, Omega, M = tf.transpose(OE[:, 0:6])

    l = M
    g = omega
    h = Omega

    L = tf.sqrt(mu * a)
    G = tf.sqrt(mu * a * (1.0 - tf.square(e)))
    H = tf.sqrt(mu * a * (1.0 - tf.square(e))) * tf.cos(i)

    DelaunayOE = tf.stack([l, g, h, L, G, H], 1)
    return DelaunayOE


def delaunay2oe_tf(DelaunayOE, mu):
    """Convert delaunay elements to traditional elements

    Args:
        trad_OE (tf.Tensor or np.array): [N x 6] (a, e, i, omega, Omega, M)

    Returns:
        delaunay_OE: [N x 6] (l, g, h, L, G, H)
    """
    l, g, h, L, G, H = tf.transpose(DelaunayOE[:, 0:6])
    Omega = h
    omega = g
    M = l
    a = tf.square(L) / mu
    e = tf.sqrt(1.0 - tf.square(G) / (mu * a))
    i = tf.acos(H / tf.sqrt(mu * a * (1.0 - tf.square(e))))
    OE = tf.stack([a, e, i, omega, Omega, M], 1)
    return OE


def cart2delaunay_tf(X, mu):
    trad = cart2trad_tf(X, mu)
    delaunay_oe = oe2delaunay_tf(trad, mu)
    return delaunay_oe


# Equinoctial


def _compute_k_cond(OE):
    I = OE[0]
    i = OE[1]
    Omega = OE[2]
    two = tf.constant(2.0, dtype=tf.float64)
    k = tf.cond(
        I == 1.0,
        lambda: tf.tan(i / 2.0) * tf.sin(Omega),
        lambda: tf.atan2(i, two) * tf.sin(Omega),
    )
    return k


def _compute_k(I, i, Omega):
    OE_subset = tf.stack([I, i, Omega], axis=1)
    k = tf.map_fn(fn=lambda OE_subset: _compute_k_cond(OE_subset), elems=OE_subset)
    return k


def _compute_h_cond(OE):
    I = OE[0]
    i = OE[1]
    Omega = OE[2]
    two = tf.constant(2.0, dtype=tf.float64)
    h = tf.cond(
        I == 1.0,
        lambda: tf.tan(i / 2.0) * tf.cos(Omega),
        lambda: tf.atan2(i, two) * tf.cos(Omega),
    )
    return h


def _compute_h(I, i, Omega):
    OE_subset = tf.stack([I, i, Omega], axis=1)
    h = tf.map_fn(fn=lambda OE_subset: _compute_h_cond(OE_subset), elems=OE_subset)
    return h


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def oe2equinoctial_tf(OE, mu):
    """Convert traditional elements to equinoctial elements

    Args:
        trad_OE: [N x 6] (a, e, i, omega, Omega, M)

    Returns:
        equi_OE (tf.Tensor or np.array): [N x 6] (p, f, g, L, h, k)
    """
    OE[:, 0]
    e = OE[:, 1]
    i = OE[:, 2]
    omega = OE[:, 3]
    Omega = OE[:, 4]
    OE[:, 5]

    # prograde i in [0, 90]
    # retrograde i in [90, 180]
    I = get_I(i)

    p = semilatus_rectum(OE)

    f = e * tf.cos(omega + I * Omega)
    g = e * tf.sin(omega + I * Omega)
    v = computeTrueAnomaly(OE)

    L = v + I * Omega + omega
    h = _compute_h(I, i, Omega)
    k = _compute_k(I, i, Omega)

    # equi_OE = tf.reshape(tf.concat([p, f, g, L, h, k], axis=0), (-1,6))
    equi_OE = tf.stack([p, f, g, L, h, k], axis=1)
    return equi_OE


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def equinoctial2oe_tf(equi_OE, mu):
    """Convert equinoctial elements to traditional elements

    Args:
        equi_OE (tf.Tensor or np.array): [N x 6] (p, f, g, L, h, k)

    Returns:
        trad_OE: [N x 6] (a, e, i, omega, Omega, M)"""
    p = equi_OE[:, 0]
    f = equi_OE[:, 1]
    g = equi_OE[:, 2]
    L = equi_OE[:, 3]
    h = equi_OE[:, 4]
    k = equi_OE[:, 5]

    e = tf.sqrt(tf.square(f) + tf.square(g))
    a = p / (1.0 - tf.square(e))

    Omega = tf.math.atan2(k, h)  # Instance 1
    # Omega = convert_angle(Omega)

    # i = 2*tf.atan2(h,tf.cos(Omega))
    one = tf.constant(1.0, dtype=tf.float64, shape=(1,))
    i = 2 * tf.atan2(tf.sqrt(tf.square(h) + tf.square(k)), one)
    I = get_I(i)

    omega = tf.math.atan2(g, f) - I * Omega
    # omega = convert_angle(omega)

    f = L - I * Omega - omega
    # f = convert_angle(f)

    M = computeMeanAnomaly(f, e)

    # trad_OE = tf.reshape(tf.concat([a, e, i, omega, Omega, M], axis=0), (-1, 6))
    trad_OE = tf.stack([a, e, i, omega, Omega, M], axis=1)
    return trad_OE


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def cart2equinoctial_tf(X, mu):
    trad = cart2trad_tf(X, mu)
    equi = oe2equinoctial_tf(trad)
    return equi


# Milankovitch
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def oe2milankovitch_tf(OE, mu):
    """Convert traditional elements to equinoctial elements

    Args:
        trad_OE: [N x 6] (a, e, i, omega, Omega, M)
        mu: gravitational parameter

    Returns:
        milankovitch_OE (tf.Tensor or np.array): [N x 7] (H1, H2, H3, e1, e2, e3, l)
    """
    cart_state = trad2cart_tf(OE, mu)
    R = cart_state[:, 0:3]
    V = cart_state[:, 3:6]
    H = tf.linalg.cross(R, V)
    eVec = tf.linalg.cross(V, H) / mu - tf.math.l2_normalize(R, axis=1)
    f = computeTrueAnomaly(OE)
    M = computeMeanAnomaly(f, tf.norm(eVec, axis=1))

    L = [OE[:, 4] + OE[:, 3] + M]
    return tf.concat((H, eVec, L), axis=1)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 7), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def milankovitch2cart_tf(milOE, mu):
    """Convert milankovitch elements to cartesian coordinates

    Args:
        milOE (tf.Tensor): [Nx7] (H1, H2, H3, e1, e2, e3, l)
        mu : Gravitational Parameters

    Returns:
        rVec, vVec: [Nx3], [Nx3] (r1, r2, r3), (v1, v2, v3)
    """
    H = milOE[:, 0:3]
    e = milOE[:, 3:6]
    L = tf.reshape(milOE[:, 6], [-1, 1])

    H_hat, H_mag = tf.linalg.normalize(H, axis=1)

    # if e_mag == 0, overwrite the e_hat vector with [1,0,0]
    e_hat = e_vec_normalization(e)
    e_mag = tf.linalg.norm(e, axis=1, keepdims=True)

    multiples = tf.convert_to_tensor([tf.shape(H)[0], 1])
    x_hat = tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=L.dtype), multiples)
    y_hat = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=L.dtype), multiples)
    z_hat = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=L.dtype), multiples)

    # (Eq. 9.150 - 9.152) Pg 558 Schaub and Junkins
    N_hat = tf.linalg.cross(H_hat, e_hat)
    omega = tf.atan2(tf_dot(e_hat, z_hat), tf_dot(N_hat, z_hat))
    Omega = tf.atan2(tf_dot(H_hat, x_hat), -tf_dot(H_hat, y_hat))

    # Need full OE to compute M
    i = tf.acos(tf_dot(H_hat, z_hat))  # if h_hat and z_hat are parallel i is undefined
    p = tf_dot(H, H) / mu

    e_squared_m1_or_1m_e_squared_value = e_squared_m1_or_1m_e_squared(e_mag)
    a = p / e_squared_m1_or_1m_e_squared_value

    M = L - Omega - omega

    OE = tf.concat([a, e_mag, i, omega, Omega, M], axis=1)
    cart_state = trad2cart_tf(OE, mu)

    return cart_state


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def cart2milankovitch_tf(X, mu):
    """
    Convert cartesian positions to milankovitch elements
    Args:
        X (tf.Tensor): [Nx6] (r1, r2, r3, v1, v2, v3)
        mu : Gravitational Parameter

    Returns:
        Milankovitch_OE: [Nx7] (H1, H2, H3, e1, e2, e3, l)
    """

    r = X[:, 0:3]
    v = X[:, 3:6]

    H = tf.linalg.cross(r, v)
    h_hat, h_mag = tf.linalg.normalize(H, axis=1)
    r_hat, r_mag = tf.linalg.normalize(r, axis=1)
    e = tf.linalg.cross(v, H) / mu - r_hat

    # Can't use tf.linalg.normalize(e, axis=1) because e_mag = 0 will produce a random e_hat vector
    # The e_vec_normalization method ensures that the value is always e_hat = [1,0,0]
    e_mag = tf.linalg.norm(e, axis=1, keepdims=True)
    e_hat = e_vec_normalization(e)

    multiples = tf.convert_to_tensor([tf.shape(H)[0], 1])
    x_hat = tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=r_mag.dtype), multiples)
    y_hat = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=r_mag.dtype), multiples)
    z_hat = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=r_mag.dtype), multiples)

    # (Eq. 9.150 - 9.152) Pg 558 Schaub and Junkins
    N_hat = tf.linalg.cross(h_hat, e_hat)
    omega = tf.atan2(tf_dot(e_hat, z_hat), tf_dot(N_hat, z_hat))
    Omega = tf.atan2(tf_dot(h_hat, x_hat), -tf_dot(h_hat, y_hat))

    # Formation Flying Slide 42
    # f = tf.atan2(tf_dot(r,N_hat),tf_dot(r,e_hat))
    f = tf.atan2(tf_dot(tf.linalg.cross(e_hat, r_hat), h_hat), tf_dot(e_hat, r_hat))
    M_val = computeMeanAnomaly(f, e_mag)
    M = tf.reshape(M_val, (-1, 1))

    # omega = convert_angle(omega)
    # Omega = convert_angle(Omega)
    # M = convert_angle(M)

    L_val = Omega + omega + M
    L = tf.reshape(L_val, (-1, 1))
    Milankovitch_OE = tf.concat((H, e, L), axis=1)
    return Milankovitch_OE


# Traditional
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def trad2cart_tf(OE, mu):
    # Pg 534, Fig 9.9 Schaub and Junkins show all possible frames
    # perifocial: i_e, i_p, i_h
    # orbit: i_r, i_theta, i_h
    # velocity frames: i_n, i_v, i_h
    OE[:, 0]
    e = OE[:, 1]
    i = OE[:, 2]
    omega = OE[:, 3]
    Omega = OE[:, 4]
    OE[:, 5]

    f = computeTrueAnomaly(OE)
    p = semilatus_rectum(OE)

    # h**2/mu = p
    h_mag = tf.sqrt(p * mu)

    cf = tf.cos(f)
    sf = tf.sin(f)

    cw = tf.cos(omega)
    sw = tf.sin(omega)

    cO = tf.cos(Omega)
    sO = tf.sin(Omega)

    ci = tf.cos(i)
    si = tf.sin(i)

    ##################
    ## New approach ##
    ##################

    # Perifocial
    # Eq. 9.45  rVec = (r cos(f) i_e + r sin(f) i_p)
    # Eq. 9.167 vVec = -mu/h * sin(f) i_e + mu/h(e + cos(f)) i_p

    # Eq. 9.46
    # NP = [
    #     [cw*cO - sw*ci*sO, -sw*cO - cw*ci*sO,  si*sO],
    #     [cw*sO + sw*ci*cO, -sw*sO + cw*ci*cO, -si*cO],
    #     [sw*si,             cw*si,             ci]
    # ]

    r_mag = p / (1.0 + e * cf)  # Eq. 9.6 Schaub and Junkins

    r_e = r_mag * (cf)  # r_vec in i_e
    r_p = r_mag * (sf)  # r_vec in i_p
    r_h = tf.zeros_like(r_e)  # r_vec in i_h

    v_e = -mu / h_mag * sf
    v_p = mu / h_mag * (e + cf)
    v_h = tf.zeros_like(v_e)

    # Eq. 9.46 (perifocal to inertial)
    r_x = (
        (cw * cO - sw * ci * sO) * r_e
        + (-sw * cO - cw * ci * sO) * r_p
        + (si * sO) * r_h
    )
    r_y = (
        (cw * sO + sw * ci * cO) * r_e
        + (-sw * sO + cw * ci * cO) * r_p
        + (-si * cO) * r_h
    )
    r_z = (sw * si) * r_e + (cw * si) * r_p + (ci) * r_h

    v_x = (
        (cw * cO - sw * ci * sO) * v_e
        + (-sw * cO - cw * ci * sO) * v_p
        + (si * sO) * v_h
    )
    v_y = (
        (cw * sO + sw * ci * cO) * v_e
        + (-sw * sO + cw * ci * cO) * v_p
        + (-si * cO) * v_h
    )
    v_z = (sw * si) * v_e + (cw * si) * v_p + (ci) * v_h

    r_xyz = tf.transpose(tf.stack([r_x, r_y, r_z], 0))
    v_xyz = tf.transpose(tf.stack([v_x, v_y, v_z], 0))

    # Perifocial conversion (Use Eqs 9.150-9.152 with 9.45)
    # Ideally we'd have e_vec, and h_vec expressed in the inertial frame
    # but such would require values of r, and v

    # ## DEBUGGING ##
    # ## CONVERT to Milankovitch directly ##
    # # ANY FRAME is inconsistent with the initial conditions!! ##
    # r_eph = tf.transpose(tf.stack([r_e, r_p, r_h], 0))
    # v_eph = tf.transpose(tf.stack([v_e, v_p, v_h], 0))

    # HVec_N = tf.linalg.cross(r_xyz, v_xyz)
    # eVec_N = tf.linalg.cross(v_xyz, HVec_N)/mu - tf.linalg.l2_normalize(r_xyz, axis=1) # This is just eVec = e*i_e

    # e_mag = tf.linalg.norm(eVec_N, keepdims=True)

    # # Eq 9.99 -- e vector in perifocal frame
    # e_e = e_mag
    # e_p = tf.zeros_like(e_e)
    # e_h = tf.zeros_like(e_e)
    # e_eph = tf.transpose(tf.stack([e_e, e_p, e_h], 0))

    # HVec_P = tf.linalg.cross(r_eph, v_eph)
    # eVec_P = tf.linalg.cross(v_eph, HVec_P)/mu - tf.linalg.l2_normalize(r_eph, axis=1) # This is just eVec = e*i_e
    # assert np.allclose(eVec_P[0],e_eph[0,0])

    # # perifocal -> cartesian
    # e_x = (cw*cO - sw*ci*sO)*e_e + (-sw*cO-cw*ci*sO)*e_p + (si*sO)*e_h
    # e_y = (cw*sO + sw*ci*cO)*e_e + (-sw*sO+cw*ci*cO)*e_p + (-si*cO)*e_h
    # e_z = (sw*si)*e_e + (cw*si)*e_p + (ci)*e_h
    # e_xyz = tf.transpose(tf.stack([e_x, e_y, e_z], 0))
    # assert np.allclose(e_xyz[0,0], eVec_N[0])

    # cT = tf.cos(omega + f)
    # sT = tf.sin(omega + f)

    # # ON (orbit - inertial) -- 9.50
    # ON = [
    #     [cT*cO - sT*ci*sO, cT*sO + sT*ci*cO, sT*si],
    #     [-sT*cO-cT*ci*sO, -sT*sO + cT*ci*cO, cT*ci],
    #     [si*sO, -si*cO, ci]
    #     ]

    # # OP (orbit - perifocal) -- 9.50
    # OP = [
    #     [ cf, sf, 0],
    #     [-sf, cf, 0],
    #     [0,    0, 1]
    #     ]

    # e_r = cf*e_e + sf*e_p + 0.0*e_h
    # e_T = -sf*e_e + cf*e_p + 0.0*e_h
    # e_h = 0*e_e + 0*e_p + 1.0*e_h

    # e_rTh = tf.transpose(tf.stack([e_r, e_T, e_h], 0))

    cart_state = tf.concat([r_xyz, v_xyz], axis=1)
    return cart_state


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ],
)
def cart2trad_tf(X, mu):
    r = X[:, 0:3]
    v = X[:, 3:6]

    h = tf.linalg.cross(r, v)
    h_hat, h_mag = tf.linalg.normalize(h, axis=1)
    p = tf_dot(h, h) / mu

    multiples = tf.convert_to_tensor([tf.shape(r)[0], 1])
    x_hat = tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=r.dtype), multiples)
    y_hat = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=r.dtype), multiples)
    z_hat = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=r.dtype), multiples)
    tf.constant(np.pi, dtype=p.dtype)
    i = tf.acos(tf_dot(h_hat, z_hat))  # if h_hat and z_hat are parallel i is undefined

    # if tf.reduce_all(i == 0) or tf.reduce_all(i == pi):
    #     raise ValueError("h_hat and z_hat are parallel (i.e. i=0 or i=pi) which prevent the system from being well defined!")

    r_hat, r_mag = tf.linalg.normalize(r, axis=1)
    e = tf.linalg.cross(v, h) / mu - r_hat

    e_mag = tf.linalg.norm(e, axis=1, keepdims=True)
    e_hat = e_vec_normalization(e)

    # Pg 558 Schaub and Junkins
    N_hat = tf.linalg.cross(h_hat, e_hat)
    omega = tf.atan2(tf_dot(e_hat, z_hat), tf_dot(N_hat, z_hat))
    Omega = tf.atan2(tf_dot(h_hat, x_hat), -tf_dot(h_hat, y_hat))

    e_squared_m1_or_1m_e_squared_value = e_squared_m1_or_1m_e_squared(e_mag)
    a = p / e_squared_m1_or_1m_e_squared_value

    # Formation Flying Slide 42 Two-Body
    f = tf.atan2(tf_dot(tf.linalg.cross(e_hat, r_hat), h_hat), tf_dot(e_hat, r_hat))
    # f = tf.atan2(tf_dot(r,N_hat),tf_dot(r,e_hat))
    M = computeMeanAnomaly(f, e_mag)

    OE = tf.transpose(
        tf.stack(
            (
                tf.squeeze(a),
                tf.squeeze(e_mag),
                tf.squeeze(i),
                tf.squeeze(omega),
                tf.squeeze(Omega),
                tf.squeeze(M),
            ),
        ),
    )
    OE = tf.reshape(OE, (-1, 6))

    return OE


# Macro functions
def oe2cart_tf(OE, mu, element_set="traditional"):
    if element_set == "traditional":
        cart_state = trad2cart_tf(OE, mu)
    elif element_set == "equinoctial":
        OE = equinoctial2oe_tf(OE)
        cart_state = trad2cart_tf(OE, mu)
    elif element_set == "milankovitch":
        cart_state = milankovitch2cart_tf(OE, mu)
    elif element_set == "delaunay":
        OE = delaunay2oe_tf(OE, mu)
        cart_state = trad2cart_tf(OE, mu)
    else:
        NotImplementedError()
    return cart_state


def cart2oe_tf(state, mu, element_set="traditional"):
    if element_set == "traditional":
        OE = cart2trad_tf(state, mu)
    elif element_set == "equinoctial":
        OE = cart2equinoctial_tf(state, mu)
    elif element_set == "milankovitch":
        OE = cart2milankovitch_tf(state, mu)
    elif element_set == "delaunay":
        OE = cart2delaunay_tf(state, mu)
    else:
        NotImplementedError()
    return OE


if __name__ == "__main__":
    import numpy as np
    from GravNN.CelestialBodies.Planets import Earth

    planet = Earth()
    mu = planet.mu
    OE = np.array([[planet.radius + 200000, 0.01, np.pi / 4, 0, 0, 0]]).astype(
        np.float32,
    )
    cart_state = trad2cart_tf(OE, mu)

    # equinoctialOE = oe2equinoctial_tf(OE)
    # OE2 = equinoctial2oe_tf(equinoctialOE)

    # delaunayOE = oe2delaunay_tf(OE,mu)
    # OE2 = delaunay2oe_tf(delaunayOE, mu)
    # assert OE2 == OE

    # equinoctialOE = oe2equinoctial_tf(OE)
    # OE2 = equinoctial2oe_tf(equinoctialOE)

    milankovitchOE = oe2milankovitch_tf(OE, mu)
    cart_state = milankovitch2cart_tf(milankovitchOE, mu)
    milOE2 = cart2milankovitch_tf(cart_state, mu)

    print("Done!")
