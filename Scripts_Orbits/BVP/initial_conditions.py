import numpy as np
from GravNN.CelestialBodies.Asteroids import Eros

from FrozenOrbits.coordinate_transforms import trad2cart_tf


def near_periodic_IC_2():
    planet = Eros()
    trad_OE = np.array(
        [[7.93e04, 1.00e-01, 1.53e00 / 2, -7.21e-01, -1.61e-01, 2.09e00]],
    )
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu)
    return trad_OE, X, T, planet


def near_periodic_IC():
    planet = Eros()
    trad_OE = np.array(
        [[7.93e04 / 2, 1.00e-01, 1.53e00 / 2, -7.21e-01, -1.61e-01, 2.09e00]],
    )
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu)
    return trad_OE, X, T, planet


def not_periodic_IC():
    planet = Eros()
    trad_OE = np.array(
        [[7.93e04 / 3, 2.00e-01, 1.53e00 / 2, -7.21e-01 * 2, -1.61e-01 * 2, 2.09e00]],
    )
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu) * 2
    return trad_OE, X, T, planet


def long_near_periodic_IC():
    planet = Eros()
    trad_OE = np.array(
        [[7.93e04 / 2, 1.00e-01, 1.53e00 / 2, -7.21e-01, -1.61e-01, 2.09e00]],
    )
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu) * 2
    return trad_OE, X, T, planet


def crazy_IC():
    planet = Eros()
    trad_OE = np.array(
        [[7.93e04 / 2.5, 3.00e-01, 1.53e00 / 4, 7.21e-01 / 3, -1.61e-01 / 2, 2.09e00]],
    )
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu)
    T *= 4
    return trad_OE, X, T, planet


def sample_initial_conditions(idx):
    planet = Eros()
    for _ in range(idx):
        a = np.random.uniform(3 * planet.radius, 7 * planet.radius)
        e = np.random.uniform(0.1, 0.3)
        i = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(0.0, 2 * np.pi)
        Omega = np.random.uniform(0.0, 2 * np.pi)
        M = np.random.uniform(0.0, 2 * np.pi)
    trad_OE = np.array([[a, e, i, omega, Omega, M]])
    X = trad2cart_tf(trad_OE, planet.mu).numpy()[0]
    T = 2 * np.pi * np.sqrt(trad_OE[0, 0] ** 3 / planet.mu)
    return trad_OE, X, T, planet
