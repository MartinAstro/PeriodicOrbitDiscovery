import numpy as np

def get_bc(element_set):
    if element_set == "traditional":
        return trad_bc
    # elif element_set == "delaunay":
    #     return dDelaunay_dt
    elif element_set == "equinoctial":
        return equi_bc
    # elif element_set == "milankovitch":
    #     return dMilankovitch_dt


def trad_bc(ya, yb, p=None):
    "Orbital elements should be equal to one another"
    "Define residuals as a 6x1 vector"
    # The mean anomaly continues to increase linearly 
    # so be sure to wrap angle by 2*pi
    yb[5] = yb[5] % (2*np.pi)

    # ensure the OE at t=0 and t=T are the same
    periodic_res = yb - ya 

    # pick the minimum distance as the residual
    if periodic_res[5] > np.pi:
        periodic_res[5] -= 2*np.pi
    elif periodic_res[5] < -np.pi:
        periodic_res[5] += 2*np.pi

    # ic_res = yb[2:5] - p
    bc_res = np.hstack((periodic_res))#, ic_res))
    return bc_res

def equi_bc(ya, yb, p=None):
    "Orbital elements should be equal to one another"
    "Define residuals as a 6x1 vector"
    # The mean anomaly continues to increase linearly 
    # so be sure to wrap angle by 2*pi
    yb[3] = yb[3] % (2*np.pi)

    # ensure the OE at t=0 and t=T are the same
    periodic_res = yb - ya 

    # pick the minimum distance as the residual
    if periodic_res[3] > np.pi:
        periodic_res[3] -= 2*np.pi
    elif periodic_res[3] < -np.pi:
        periodic_res[3] += 2*np.pi

    # ic_res = yb[2:5] - p
    bc_res = np.hstack((periodic_res))#, ic_res))
    return bc_res