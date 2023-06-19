import numpy as np
import tensorflow as tf

from FrozenOrbits.coordinate_transforms import *


def element_dot(a, b):
    # what once was np.dot
    return tf.reshape(tf.reduce_sum(a * b, axis=1), ((-1, 1)))


class LPE_Milankovitch:
    def __init__(self, model, mu, l_star=1.0, t_star=1.0, m_star=1.0):
        self.model = model
        self.G_tilde = 6.67408 * 10e-11  # m^3/(kg*s^2)
        self.mu_tilde = tf.constant(mu, dtype=tf.float64, name="mu")
        self.l_star = tf.constant(l_star, dtype=tf.float64, name="ref_length")
        self.m_star = tf.constant(m_star, dtype=tf.float64, name="ref_mass")
        self.t_star = tf.constant(t_star, dtype=tf.float64, name="ref_time")

        self.mu = self.mu_tilde * (self.t_star**2 / self.l_star**3)
        self.element_set = "milankovitch"
        self.num_elements = 7

    def non_dimensionalize_state(self, x0):
        mil_OE_ND = tf.concat(
            [(self.t_star / self.l_star**2) * x0[:, 0:3], x0[:, 3:7]],
            axis=1,
        )
        return mil_OE_ND

    def dimensionalize_state(self, x0):
        mil_OE = tf.concat(
            [(self.l_star**2 / self.t_star) * x0[:, 0:3], x0[:, 3:7]],
            axis=1,
        )
        return mil_OE

    def non_dimensionalize_time(self, T):
        T_ND = T / self.t_star
        return T_ND

    def dimensionalize_time(self, T):
        T = T * self.t_star
        return T

    def _conditional_compute_n(self, a):
        n = tf.cond(
            tf.greater_equal(a, 0),
            lambda: tf.sqrt(self.mu / a**3),
            lambda: tf.sqrt(-self.mu / a**3),
        )
        return n

    def _compute_n(self, a):
        n = tf.map_fn(fn=lambda a: self._conditional_compute_n(a), elems=(a))
        return n

    def dOE_dt(self, mil_OE):
        mil_OE_input = mil_OE.reshape((1, -1)).astype(np.float64)
        mil_OE_input_tf = tf.Variable(
            mil_OE_input,
            dtype=tf.float64,
            name="orbit_elements",
        )
        dOE_dt_results = self.dOE_milankovitch_dt(mil_OE_input_tf, self.mu)
        return dOE_dt_results.numpy()

    def dOE_dt_dx(self, mil_OE):
        mil_OE_input = mil_OE.reshape((1, -1)).astype(np.float64)
        mil_OE_input_tf = tf.Variable(
            mil_OE_input,
            dtype=tf.float64,
            name="orbit_elements",
        )
        dOE_dt_dx_results = self.dOE_milankovitch_dt_dx(mil_OE_input_tf, self.mu)
        return dOE_dt_dx_results.numpy()

    def generate_potential(self, x):
        """
        Very silly function, but it's the only way to compile dOE_milankovitch_dt_dx as a tf.function().
        In short, tf.function is unable to accept the model as an argument without constant retracing
        according to: https://github.com/tensorflow/tensorflow/issues/38875
        """
        U_tilde = self.model.generate_potential_tf(x)
        return U_tilde

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 7), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ],
    )
    def dOE_milankovitch_dt_dx(self, milankovitch_OE, mu):
        print("Retracing")
        H = milankovitch_OE[:, 0:3]
        e = milankovitch_OE[:, 3:6]
        L = milankovitch_OE[:, 6]
        with tf.GradientTape(persistent=True) as tape_dx:
            tape_dx.watch(milankovitch_OE)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(milankovitch_OE)
                mil_OE_dim = self.dimensionalize_state(milankovitch_OE)
                cart_state = milankovitch2cart_tf(mil_OE_dim, mu)
                r_tilde = cart_state[:, 0:3]
                u_pred_tilde = -self.generate_potential(r_tilde)
                U = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
            dUdOE = tape.gradient(U, milankovitch_OE)

            dUdH = dUdOE[:, 0:3]
            dUde = dUdOE[:, 3:6]
            dUdl = dUdOE[:, 6]

            p = element_dot(H, H) / mu
            e_squared = element_dot(e, e)
            denominator = tf.cond(
                e_squared >= 1.0,
                lambda: (e_squared - 1.0),
                lambda: (1.0 - e_squared),
            )
            a = p / denominator
            n = self._compute_n(a)

            r = r_tilde / self.l_star

            H_mag = tf.norm(H)
            tf.norm(e)
            tf.norm(r)

            multiples = tf.convert_to_tensor([tf.shape(H)[0], 1])
            z_hat = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float64), multiples)

            HVec = H
            eVec = e
            tf.reshape(L, (-1, 1))

            term1 = (HVec + H_mag * z_hat) / (H_mag + element_dot(z_hat, HVec))
            term2 = denominator / H_mag**2
            term6 = (
                tf.sqrt(denominator) / (tf.sqrt(mu * a) + H_mag) * eVec
                + tf_dot(z_hat, eVec) / (H_mag * (H_mag + tf_dot(z_hat, HVec))) * HVec
            )

            H_dt = (1.0 / self.m_star) * (
                tf.linalg.cross(HVec, dUdH) + tf.linalg.cross(eVec, dUde) + term1 * dUdl
            )
            e_dt = (1.0 / self.m_star) * (
                tf.linalg.cross(eVec, dUdH)
                + term2 * tf.linalg.cross(HVec, dUde)
                - term6 * dUdl
            )
            L_dt = (1.0 / self.m_star) * (
                -element_dot(term1, dUdH) + element_dot(term6, dUde)
            ) + n

            dOE_dt = tf.concat([H_dt, e_dt, L_dt], axis=1)
        dOE_dt_dx = tape_dx.batch_jacobian(
            dOE_dt,
            milankovitch_OE,
            experimental_use_pfor=False,
        )
        return dOE_dt_dx[0]

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 7), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
        ],
    )
    def dOE_milankovitch_dt(self, milankovitch_OE, mu):
        H = milankovitch_OE[:, 0:3]
        e = milankovitch_OE[:, 3:6]
        L = milankovitch_OE[:, 6]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(milankovitch_OE)
            mil_OE_dim = self.dimensionalize_state(milankovitch_OE)
            cart_state = milankovitch2cart_tf(mil_OE_dim, self.mu_tilde)
            r_tilde = cart_state[:, 0:3]
            u_pred_tilde = -self.generate_potential(r_tilde)
            U = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
        dUdOE = tape.gradient(U, milankovitch_OE)

        p = element_dot(H, H) / mu  # TODO: non-dimensionalize mu
        e_squared = element_dot(e, e)
        denominator = tf.cond(
            e_squared >= 1.0,
            lambda: (e_squared - 1),
            lambda: (1 - e_squared),
        )
        a = p / denominator
        n = self._compute_n(a)

        dUdH = dUdOE[:, 0:3]
        dUde = dUdOE[:, 3:6]
        dUdl = dUdOE[:, 6]

        r = r_tilde / self.l_star

        H_mag = tf.norm(H, axis=1)
        tf.norm(e, axis=1)
        tf.norm(r, axis=1)

        multiples = tf.convert_to_tensor([tf.shape(H)[0], 1])
        z_hat = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float64), multiples)

        HVec = H
        eVec = e
        tf.reshape(L, (-1, 1))

        term1 = (HVec + H_mag * z_hat) / (H_mag + element_dot(z_hat, HVec))
        term2 = denominator / H_mag**2
        term6 = (
            tf.sqrt(denominator) / (tf.sqrt(mu * a) + H_mag) * eVec
            + tf_dot(z_hat, eVec) / (H_mag * (H_mag + tf_dot(z_hat, HVec))) * HVec
        )

        H_dt = (1.0 / self.m_star) * (
            tf.linalg.cross(HVec, dUdH) + tf.linalg.cross(eVec, dUde) + term1 * dUdl
        )
        e_dt = (1.0 / self.m_star) * (
            tf.linalg.cross(eVec, dUdH)
            + term2 * tf.linalg.cross(HVec, dUde)
            - term6 * dUdl
        )
        L_dt = (1.0 / self.m_star) * (
            -element_dot(term1, dUdH) + element_dot(term6, dUde)
        ) + n

        dOE_dt = tf.concat([H_dt, e_dt, L_dt], axis=1)

        return dOE_dt[0]


class LPE_Traditional:
    "LPE with traditional OE, but normalized by length, time, and mass"

    def __init__(self, model, mu, l_star=1.0, t_star=1.0, m_star=1.0, theta_star=1.0):
        self.model = model
        self.mu_tilde = tf.constant(mu, dtype=tf.float64, name="mu")
        self.l_star = tf.constant(l_star, dtype=tf.float64, name="ref_length")
        self.m_star = tf.constant(m_star, dtype=tf.float64, name="ref_mass")
        self.t_star = tf.constant(t_star, dtype=tf.float64, name="ref_time")
        self.theta_star = tf.constant(theta_star, dtype=tf.float64, name="ref_angle")
        self.mu = self.mu_tilde * (self.t_star**2 / self.l_star**3)
        self.element_set = "traditional"
        self.num_elements = 6

    def non_dimensionalize_state(self, x0):
        a = tf.reshape(1.0 / self.l_star * x0[:, 0], (-1, 1))
        trad_OE_ND = tf.concat([a, x0[:, 1:]], axis=1)
        return trad_OE_ND

    def dimensionalize_state(self, x0):
        a_tilde = tf.reshape(self.l_star * x0[:, 0], (-1, 1))
        trad_OE = tf.concat([a_tilde, x0[:, 1:]], axis=1)
        return trad_OE

    def non_dimensionalize_time(self, T):
        T_ND = T / self.t_star
        return T_ND

    def dimensionalize_time(self, T_ND):
        T = T_ND * self.t_star
        return T

    def dOE_dt(self, OE):
        OE_input = OE.reshape((1, -1)).astype(np.float64)
        OE_input_tf = tf.Variable(OE_input, dtype=tf.float64, name="orbit_elements")
        dOE_dt_results = self.dOE_trad_dt(OE_input_tf, self.mu)
        return dOE_dt_results.numpy()[0]

    def dOE_dt_dx(self, OE):
        OE_input = OE.reshape((1, -1)).astype(np.float64)
        OE_input_tf = tf.Variable(OE_input, dtype=tf.float64, name="orbit_elements")
        dOE_dt_dx_results = self.dOE_trad_dt_dx(OE_input_tf, self.mu)
        return dOE_dt_dx_results.numpy()[0]

    def generate_potential(self, x):
        """
        Very silly function, but it's the only way to compile dOE_dt_dx as a tf.function().
        In short, tf.function is unable to accept the model as an argument without constant retracing
        according to: https://github.com/tensorflow/tensorflow/issues/38875
        """
        return self.model.generate_potential_tf(x)

    def _conditional_compute_b(self, OE):
        a = OE[0]
        e = OE[1]
        b = tf.cond(
            tf.greater_equal(tf.abs(e), 1),
            lambda: a * tf.sqrt(e**2 - 1.0),
            lambda: a * tf.sqrt(1.0 - e**2),
        )
        return b

    def _conditional_compute_n(self, a):
        n = tf.cond(
            tf.greater_equal(a, 0),
            lambda: tf.sqrt(self.mu / a**3),
            lambda: tf.sqrt(-self.mu / a**3),
        )
        return n

    def _compute_b(self, OE):
        b = tf.map_fn(fn=lambda OE: self._conditional_compute_b(OE), elems=OE)
        return b

    def _compute_n(self, a):
        n = tf.map_fn(fn=lambda a: self._conditional_compute_n(a), elems=(a))
        return n

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ],
    )
    def dOE_trad_dt(self, OE, mu):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE)
            trad_OE_dim = self.dimensionalize_state(OE)
            cart_state = oe2cart_tf(trad_OE_dim, self.mu_tilde)
            r_tilde = cart_state[:, 0:3]
            u_pred_tilde = -self.generate_potential(r_tilde)  # THIS MUST BE NEGATED
            u_pred = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
        dUdOE = tape.gradient(u_pred, OE)

        a = OE[:, 0]
        e = OE[:, 1]
        i = OE[:, 2]  # transpose necessary to assign

        b = self._compute_b(OE)
        n = self._compute_n(a)

        # Schaub 12.85 or
        # https://farside.ph.utexas.edu/teaching/celestial/Celestial/node155.html
        dadt = (1.0 / self.m_star) * (2.0 / (n * a) * dUdOE[:, 5])
        dedt = (1.0 / self.m_star) * (
            -b / (n * a**3 * e) * dUdOE[:, 3]
            + b**2 / (n * a**4 * e) * dUdOE[:, 5]
        )
        didt = (1.0 / self.m_star) * (
            -1.0 / (n * a * b * tf.sin(i)) * dUdOE[:, 4]
            + tf.cos(i) / (n * a * b * tf.sin(i)) * dUdOE[:, 3]
        )
        domegadt = (1.0 / self.m_star) * (
            -tf.cos(i) / (n * a * b * tf.sin(i)) * dUdOE[:, 2]
            + b / (n * a**3 * e) * dUdOE[:, 1]
        )
        dOmegadt = (1.0 / self.m_star) * (1.0 / (n * a * b * tf.sin(i)) * dUdOE[:, 2])
        dMdt = n - (1.0 / self.m_star) * (
            2.0 / (n * a) * dUdOE[:, 0]
            + (1.0 - e**2) / (n * a**2 * e) * dUdOE[:, 1]
        )

        dOE_dt = tf.reshape(
            tf.concat([dadt, dedt, didt, domegadt, dOmegadt, dMdt], axis=0),
            (-1, 6),
        )

        return dOE_dt

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ],
    )
    def dOE_trad_dt_dx(self, OE, mu):
        """For some reason, cannot use self.dOE_trad_dt() within this function, but instead the whole
        contents must be copied instead."""
        print("Retracing")
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE)
            with tf.GradientTape(persistent=True) as tape_dt:
                tape_dt.watch(OE)
                trad_OE_dim = self.dimensionalize_state(OE)
                cart_state = oe2cart_tf(trad_OE_dim, self.mu_tilde)
                r_tilde = cart_state[:, 0:3]
                u_pred_tilde = -self.generate_potential(r_tilde)
                u_pred = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
            dUdOE = tape_dt.gradient(u_pred, OE)

            a = OE[:, 0]
            e = OE[:, 1]
            i = OE[:, 2]  # transpose necessary to assign

            b = self._compute_b(OE)
            n = self._compute_n(a)

            dadt = (1.0 / self.m_star) * (2.0 / (n * a) * dUdOE[:, 5])
            dedt = (1.0 / self.m_star) * (
                -b / (n * a**3 * e) * dUdOE[:, 3]
                + b**2 / (n * a**4 * e) * dUdOE[:, 5]
            )
            didt = (1.0 / self.m_star) * (
                -1.0 / (n * a * b * tf.sin(i)) * dUdOE[:, 4]
                + tf.cos(i) / (n * a * b * tf.sin(i)) * dUdOE[:, 3]
            )
            domegadt = (1.0 / self.m_star) * (
                -tf.cos(i) / (n * a * b * tf.sin(i)) * dUdOE[:, 2]
                + b / (n * a**3 * e) * dUdOE[:, 1]
            )
            dOmegadt = (1.0 / self.m_star) * (
                1.0 / (n * a * b * tf.sin(i)) * dUdOE[:, 2]
            )
            dMdt = n - (1.0 / self.m_star) * (
                2.0 / (n * a) * dUdOE[:, 0]
                + (1 - e**2) / (n * a**2 * e) * dUdOE[:, 1]
            )

            dOE_dt = tf.reshape(
                tf.concat([dadt, dedt, didt, domegadt, dOmegadt, dMdt], axis=0),
                (-1, 6),
            )

        dOE_dt_dx = tape.batch_jacobian(dOE_dt, OE, experimental_use_pfor=False)
        return dOE_dt_dx


class LPE_Cartesian:
    "LPE with cartesian coordinates, but normalized by length, time, and mass"

    def __init__(self, model, mu, l_star=1.0, t_star=1.0, m_star=1.0):
        self.model = model
        self.mu_tilde = tf.constant(mu, dtype=tf.float64, name="mu")
        self.l_star = tf.constant(l_star, dtype=tf.float64, name="ref_length")
        self.m_star = tf.constant(m_star, dtype=tf.float64, name="ref_mass")
        self.t_star = tf.constant(t_star, dtype=tf.float64, name="ref_time")
        self.mu = self.mu_tilde * (self.t_star**2 / self.l_star**3)
        self.element_set = "traditional"
        self.num_elements = 6

    def non_dimensionalize_state(self, x0):
        x_ND = tf.reshape(1.0 / self.l_star * x0[:, 0:3], (-1, 3))
        v_ND = tf.reshape(self.t_star / self.l_star * x0[:, 3:6], (-1, 3))
        cart_ND = tf.concat([x_ND, v_ND], axis=1)
        return cart_ND

    def dimensionalize_state(self, x0):
        x = tf.reshape(self.l_star * x0[:, 0:3], (-1, 3))
        v = tf.reshape(self.l_star / self.t_star * x0[:, 3:6], (-1, 3))
        cart = tf.concat([x, v], axis=1)
        return cart

    def non_dimensionalize_time(self, T):
        T_ND = T / self.t_star
        return T_ND

    def dimensionalize_time(self, T_ND):
        T = T_ND * self.t_star
        return T

    def dOE_dt(self, OE):
        OE_input = OE.reshape((1, -1)).astype(np.float64)
        OE_input_tf = tf.Variable(OE_input, dtype=tf.float64, name="orbit_elements")
        dOE_dt_results = self.dOE_cart_dt(OE_input_tf, self.mu)
        return dOE_dt_results.numpy()[0]

    def dOE_dt_dx(self, OE):
        OE_input = OE.reshape((1, -1)).astype(np.float64)
        OE_input_tf = tf.Variable(OE_input, dtype=tf.float64, name="orbit_elements")
        dOE_dt_dx_results = self.dOE_cart_dt_dx(OE_input_tf, self.mu)
        return dOE_dt_dx_results.numpy()[0]

    def generate_potential(self, x):
        """
        Very silly function, but it's the only way to compile dOE_dt_dx as a tf.function().
        In short, tf.function is unable to accept the model as an argument without constant retracing
        according to: https://github.com/tensorflow/tensorflow/issues/38875
        """
        return self.model.generate_potential_tf(x)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ],
    )
    def dOE_cart_dt(self, OE, mu):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE)
            trad_OE_dim = self.dimensionalize_state(OE)
            r_tilde = trad_OE_dim[:, 0:3]
            u_0 = -self.mu_tilde / tf.linalg.norm(r_tilde, keepdims=True)
            u_pred_tilde = u_0 + self.generate_potential(
                r_tilde,
            )  # THIS MUST BE NEGATED
            u_pred = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
        a_tilde = -tape.gradient(u_pred, OE)
        v_tilde = OE[:, 3:6]
        dOE_dt = tf.reshape(
            tf.concat(
                [
                    v_tilde[:, 0],
                    v_tilde[:, 1],
                    v_tilde[:, 2],
                    a_tilde[:, 0],
                    a_tilde[:, 1],
                    a_tilde[:, 2],
                ],
                axis=0,
            ),
            (-1, 6),
        )

        return dOE_dt

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ],
    )
    def dOE_cart_dt_dx(self, OE, mu):
        """For some reason, cannot use self.dOE_trad_dt() within this function, but instead the whole
        contents must be copied instead."""
        print("Retracing")
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE)
            with tf.GradientTape(persistent=True) as tape_dt:
                tape_dt.watch(OE)
                trad_OE_dim = self.dimensionalize_state(OE)
                r_tilde = trad_OE_dim[:, 0:3]
                u_0 = -self.mu_tilde / tf.linalg.norm(r_tilde, keepdims=True)
                u_pred_tilde = u_0 + self.generate_potential(
                    r_tilde,
                )  # THIS MUST BE NEGATED
                u_pred = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
            a_tilde = -tape_dt.gradient(u_pred, OE)
            v_tilde = OE[:, 3:6]
            dOE_dt = tf.reshape(
                tf.concat(
                    [
                        v_tilde[:, 0],
                        v_tilde[:, 1],
                        v_tilde[:, 2],
                        a_tilde[:, 0],
                        a_tilde[:, 1],
                        a_tilde[:, 2],
                    ],
                    axis=0,
                ),
                (-1, 6),
            )

        dOE_dt_dx = tape.batch_jacobian(dOE_dt, OE, experimental_use_pfor=False)
        return dOE_dt_dx


class LPE_Equinoctial:
    def __init__(self, model, mu, l_star=1.0, t_star=1.0, m_star=1.0):
        self.model = model
        self.G_tilde = 6.67408 * 10e-11  # m^3/(kg*s^2)
        self.mu_tilde = tf.constant(mu, dtype=tf.float64, name="mu")
        self.l_star = tf.constant(l_star, dtype=tf.float64, name="ref_length")
        self.m_star = tf.constant(m_star, dtype=tf.float64, name="ref_mass")
        self.t_star = tf.constant(t_star, dtype=tf.float64, name="ref_time")

        self.mu = self.mu_tilde * (self.t_star**2 / self.l_star**3)
        self.element_set = "equinoctial"
        self.num_elements = 6

    def non_dimensionalize_state(self, x0):
        p_non_dim = tf.reshape(x0[:, 0] / self.l_star, (-1, 1))
        equi_OE_ND = tf.concat([p_non_dim, x0[:, 1:]], axis=1)
        return equi_OE_ND

    def dimensionalize_state(self, x0):
        p_dim = tf.reshape(x0[:, 0] * self.l_star, (-1, 1))
        equi_OE = tf.concat([p_dim, x0[:, 1:]], axis=1)
        return equi_OE

    def non_dimensionalize_time(self, T):
        T_ND = T / self.t_star
        return T_ND

    def dimensionalize_time(self, T):
        T = T * self.t_star
        return T

    def _conditional_compute_n(self, a):
        n = tf.cond(
            tf.greater_equal(a, 0),
            lambda: tf.sqrt(self.mu / a**3),
            lambda: tf.sqrt(-self.mu / a**3),
        )
        return n

    def _compute_n(self, a):
        n = tf.map_fn(fn=lambda a: self._conditional_compute_n(a), elems=(a))
        return n

    def dOE_dt(self, equi_OE):
        equi_OE_input = equi_OE.reshape((1, -1)).astype(np.float64)
        equi_OE_input_tf = tf.Variable(
            equi_OE_input,
            dtype=tf.float64,
            name="orbit_elements",
        )
        dOE_dt_results = self.dOE_equinoctial_dt(equi_OE_input_tf, self.mu)
        return dOE_dt_results.numpy()

    def dOE_dt_dx(self, equi_OE):
        equi_OE_input = equi_OE.reshape((1, -1)).astype(np.float64)
        equi_OE_input_tf = tf.Variable(
            equi_OE_input,
            dtype=tf.float64,
            name="orbit_elements",
        )
        dOE_dt_dx_results = self.dOE_equinoctial_dt_dx(equi_OE_input_tf, self.mu)
        return dOE_dt_dx_results.numpy()

    def generate_potential(self, x):
        """
        Very silly function, but it's the only way to compile dOE_equinoctial_dt_dx as a tf.function().
        In short, tf.function is unable to accept the model as an argument without constant retracing
        according to: https://github.com/tensorflow/tensorflow/issues/38875
        """
        U_tilde = self.model.generate_potential_tf(x)
        return U_tilde

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ],
    )
    def dOE_equinoctial_dt_dx(self, equinoctial_OE, mu):
        print("Retracing")

        with tf.GradientTape(persistent=True) as tape_dx:
            tape_dx.watch(equinoctial_OE)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(equinoctial_OE)
                equi_OE_dim = self.dimensionalize_state(equinoctial_OE)
                trad_OE = equinoctial2oe_tf(equi_OE_dim, self.mu_tilde)
                cart_state = oe2cart_tf(trad_OE, self.mu_tilde)
                r_tilde = cart_state[:, 0:3]
                u_pred_tilde = -self.generate_potential(r_tilde)
                U = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
            dUdOE = tape.gradient(U, equinoctial_OE)

            p = equinoctial_OE[:, 0]
            f = equinoctial_OE[:, 1]
            g = equinoctial_OE[:, 2]
            L = equinoctial_OE[:, 3]
            h = equinoctial_OE[:, 4]
            k = equinoctial_OE[:, 5]

            s = tf.sqrt(1.0 + h**2 + k**2)
            w = 1.0 + f * tf.cos(L) + g * tf.sin(L)
            # p, f, g, L, h, k

            dpdt = (
                2.0
                * tf.sqrt(p / mu)
                * (-g * dUdOE[:, 1] + f * dUdOE[:, 2] + dUdOE[:, 3]),
            )
            dfdt = (
                1.0
                / tf.sqrt(mu * p)
                * (
                    2.0 * p * g * dUdOE[:, 0]
                    - (1.0 - f**2 - g**2) * dUdOE[:, 2]
                    - g * s**2 / 2.0 * (h * dUdOE[:, 4] + k * dUdOE[:, 5])
                    + (f + (1.0 + w) * tf.cos(L)) * dUdOE[:, 3]
                ),
            )
            dgdt = (
                1.0
                / tf.sqrt(mu * p)
                * (
                    -2.0 * p * f * dUdOE[:, 0]
                    + (1.0 - f**2 - g**2) * dUdOE[:, 1]
                    + f * s**2 / 2.0 * (h * dUdOE[:, 4] + k * dUdOE[:, 5])
                    + (g + (1.0 + w) * tf.sin(L)) * dUdOE[:, 3]
                ),
            )
            dLdt = (
                tf.sqrt(mu * p) * (w / p) ** 2
                + s**2
                / (2.0 * tf.sqrt(mu * p))
                * (h * dUdOE[:, 4] + k * dUdOE[:, 5]),
            )
            dhdt = (
                s**2
                / (2.0 * tf.sqrt(mu * p))
                * (
                    h * (g * dUdOE[:, 1] - f * dUdOE[:, 2] - dUdOE[:, 3])
                    - s**2 / 2.0 * dUdOE[:, 5]
                ),
            )
            dkdt = (
                s**2
                / (2.0 * tf.sqrt(mu * p))
                * (
                    k * (g * dUdOE[:, 1] - f * dUdOE[:, 2] - dUdOE[:, 3])
                    + s**2 / 2.0 * dUdOE[:, 4]
                )
            )
            dOE_dt = tf.concat(
                [dpdt, dfdt, dgdt, dLdt, dhdt, tf.reshape(dkdt, (-1, 1))],
                axis=1,
            )
        dOE_dt_dx = tape_dx.batch_jacobian(
            dOE_dt,
            equinoctial_OE,
            experimental_use_pfor=False,
        )
        return dOE_dt_dx[0]

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 6), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
        ],
    )
    def dOE_equinoctial_dt(self, equinoctial_OE, mu):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(equinoctial_OE)
            equi_OE_dim = self.dimensionalize_state(equinoctial_OE)
            trad_OE = equinoctial2oe_tf(equi_OE_dim, self.mu_tilde)
            cart_state = oe2cart_tf(trad_OE, self.mu_tilde)
            r_tilde = cart_state[:, 0:3]
            u_pred_tilde = -self.generate_potential(r_tilde)
            U = (self.t_star / self.l_star) ** 2 * self.m_star * u_pred_tilde
        dUdOE = tape.gradient(U, equinoctial_OE)

        p = equinoctial_OE[:, 0]
        f = equinoctial_OE[:, 1]
        g = equinoctial_OE[:, 2]
        L = equinoctial_OE[:, 3]
        h = equinoctial_OE[:, 4]
        k = equinoctial_OE[:, 5]

        s = tf.sqrt(1.0 + h**2 + k**2)
        w = 1.0 + f * tf.cos(L) + g * tf.sin(L)
        # p, f, g, L, h, k

        dpdt = (
            2.0 * tf.sqrt(p / mu) * (-g * dUdOE[:, 1] + f * dUdOE[:, 2] + dUdOE[:, 3]),
        )
        dfdt = (
            1.0
            / tf.sqrt(mu * p)
            * (
                2.0 * p * g * dUdOE[:, 0]
                - (1.0 - f**2 - g**2) * dUdOE[:, 2]
                - g * s**2 / 2.0 * (h * dUdOE[:, 4] + k * dUdOE[:, 5])
                + (f + (1.0 + w) * tf.cos(L)) * dUdOE[:, 3]
            ),
        )
        dgdt = (
            1.0
            / tf.sqrt(mu * p)
            * (
                -2.0 * p * f * dUdOE[:, 0]
                + (1.0 - f**2 - g**2) * dUdOE[:, 1]
                + f * s**2 / 2.0 * (h * dUdOE[:, 4] + k * dUdOE[:, 5])
                + (g + (1.0 + w) * tf.sin(L)) * dUdOE[:, 3]
            ),
        )
        dLdt = (
            tf.sqrt(mu * p) * (w / p) ** 2
            + s**2 / (2.0 * tf.sqrt(mu * p)) * (h * dUdOE[:, 4] + k * dUdOE[:, 5]),
        )
        dhdt = (
            s**2
            / (2.0 * tf.sqrt(mu * p))
            * (
                h * (g * dUdOE[:, 1] - f * dUdOE[:, 2] - dUdOE[:, 3])
                - s**2 / 2.0 * dUdOE[:, 5]
            ),
        )
        dkdt = (
            s**2
            / (2.0 * tf.sqrt(mu * p))
            * (
                k * (g * dUdOE[:, 1] - f * dUdOE[:, 2] - dUdOE[:, 3])
                + s**2 / 2.0 * dUdOE[:, 4]
            )
        )

        dOE_dt = tf.concat(
            [dpdt, dfdt, dgdt, dLdt, dhdt, tf.reshape(dkdt, (-1, 1))],
            axis=1,
        )

        return dOE_dt[0]
