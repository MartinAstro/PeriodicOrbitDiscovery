import numpy as np
from FrozenOrbits.coordinate_transforms import *
from GravNN.CelestialBodies.Planets import Earth


def test_traditional():
    planet = Earth()
    OE = np.array([[planet.radius + 200000, 0.2, np.pi/4, np.pi/4, np.pi/4, np.pi/4]])
    cart_state = trad2cart_tf(OE, planet.mu)
    OE2 = cart2trad_tf(cart_state, planet.mu)
    assert np.allclose(OE2, OE, rtol=1E-14)

def test_milankovitch():
    planet = Earth()
    # milankovitch_OE = np.array([[10000, 10000, 10000, 0.1, 0.3, 0.2, np.pi/4]])
    milankovitch_OE = np.array([[1E12, 1E12, 1E12, 0.1, 0.3, 0.2, np.pi/4]])
    # milankovitch_OE = np.array([[1E12, 1E10, 1E13, 0.1, 0.3, 0.2, np.pi/4]])
    cart_state = milankovitch2cart_tf(milankovitch_OE, planet.mu)
    mil_OE2 = cart2milankovitch_tf(cart_state, planet.mu)
    assert np.allclose(mil_OE2, milankovitch_OE, rtol=1E-14), f"Milankovitch conversion not conserved \n OE1 {milankovitch_OE}\n OE2 {mil_OE2}"

# def test_equinoctial():
#     planet = Earth()
#     equinoctial_OE = np.array([[10000, 10000, 10000, 0.1, 0.3, 0.2, np.pi/4]])
#     cart_state = equinoctial2cart_tf(equinoctial_OE, planet.mu)
#     equinoctial_OE_2 = cart2equinoctial_tf(cart_state, planet.mu)
#     assert np.allclose(equinoctial_OE, equinoctial_OE_2, rtol=1E-14)

# def test_delaunay():
#     planet = Earth()
#     delaunay_OE = np.array([[10000, 10000, 10000, 0.1, 0.3, 0.2, np.pi/4]])
#     cart_state = delaunay2cart_tf(delaunay_OE, planet.mu)
#     delaunay_OE_2 = cart2delaunay_tf(cart_state, planet.mu)
#     assert np.allclose(delaunay_OE, delaunay_OE_2, rtol=1E-14)




def main():
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    # test_traditional()
    test_milankovitch()

if __name__ == "__main__":
    main()