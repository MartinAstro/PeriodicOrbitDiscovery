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
    mil_OE1 = np.array([[1E12, 1E12, 1E12, 0.1, 0.3, 0.2, np.pi/4]])
    cart_state_1 = milankovitch2cart_tf(mil_OE1, planet.mu)
    mil_OE2 = cart2milankovitch_tf(cart_state_1, planet.mu)

    # NOTE: Because the milankovitch elements are a redundant set, once a set is mapped to the 
    # state vector, there isn't a guarantee that the state vector can be mapped to the original
    # element set. This prevents us from using the following assert: 

    # assert np.allclose(mil_OE2, mil_OE1, rtol=1E-14), \
    #     f"Milankovitch conversion not conserved \n OE1 {mil_OE1}\n OE2 {mil_OE2}"

    # Instead, we can repeat the mapping from the new element set back to cartesian, and
    # confirm that the state vectors are equivalent.

    cart_state_2 = milankovitch2cart_tf(mil_OE2, planet.mu)
    assert np.allclose(cart_state_1, cart_state_2)
    
    # Note, once the element set has been mapped from OE -> X -> OE', if the conversion
    # between OE' -> X -> OE'' occurs, you will find that OE' == OE''. This is because
    # the conversion produces an element set self consistent with original transformation.

    mil_OE3 = cart2milankovitch_tf(cart_state_2, planet.mu)
    assert np.allclose(mil_OE2, mil_OE3, rtol=1E-14), \
        f"Milankovitch conversion not conserved \n OE1 {mil_OE3}\n OE2 {mil_OE2}"




def test_milankovitch_circular():
    planet = Earth()
    milankovitch_OE = np.array([[1E12, 1E12, 1E12, 0.0, 0.0, 0.0, np.pi/4]])
    cart_state = milankovitch2cart_tf(milankovitch_OE, planet.mu)
    OE = cart2trad_tf(cart_state, planet.mu)
    mil_OE2 = oe2milankovitch_tf(OE, planet.mu)
    assert np.allclose(mil_OE2, milankovitch_OE, rtol=1E-14), \
        f"Milankovitch conversion not conserved \n OE1 {milankovitch_OE}\n OE2 {mil_OE2}"

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
    # test_milankovitch_circular()

if __name__ == "__main__":
    main()