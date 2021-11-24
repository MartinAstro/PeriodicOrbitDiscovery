import numpy as np

def fine_periodic(t, y, IC):
    R, V = y[0:3], y[3:6]
    R0, V0 = IC[0:3], IC[3:6]

    dR = np.linalg.norm(R0 - R)
    dV = np.linalg.norm(V0 - V)

    near_periodic = (dR-500) + (dV - 1)
    return near_periodic

def coarse_periodic(t, y, IC):
    R, V = y[0:3], y[3:6]
    R0, V0 = IC[0:3], IC[3:6]

    dR = np.linalg.norm(R0 - R)
    dV = np.linalg.norm(V0 - V)

    near_periodic = (dR-1000) + (dV - 1)
    return near_periodic

def very_coarse_periodic(t, y, IC):
    R, V = y[0:3], y[3:6]
    R0, V0 = IC[0:3], IC[3:6]

    dR = np.linalg.norm(R0 - R)
    dV = np.linalg.norm(V0 - V)
    
    near_periodic = (dR-5000) + (dV - 1)
    return near_periodic
