import numpy as np
from FrozenOrbits.coordinate_transforms import cart2oe_tf
def compute_period(mu, a):
    if a > 0.0:
        n = np.sqrt(mu/a**3)
    else:
        n = np.sqrt(mu/(-a**3))
    T = 2*np.pi/n 
    return T

def compute_period_from_state(mu, state):
    oe = cart2oe_tf(state, mu).numpy()
    T = compute_period(mu, oe[0,0])
    return T

def sample_safe_trad_OE(R_min, R_max):
    a = np.random.uniform(R_min, R_max)
    e_1 = 1. - R_min/a
    e_2 = R_max/a - 1.
    e = np.random.uniform(0, np.min([e_1, e_2]))

    trad_OE = np.array([[a, 
                        e, 
                        np.random.uniform(0.0, np.pi),
                        np.random.uniform(0.0, 2*np.pi),
                        np.random.uniform(0.0, 2*np.pi),
                        np.random.uniform(0.0, 2*np.pi)]]) 
    return trad_OE
