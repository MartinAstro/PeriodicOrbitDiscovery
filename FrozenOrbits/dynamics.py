import numpy as np
def dynamics_cart(t, x, model, pbar): 
    r = x[0:3].reshape((1, -1))
    a = model.generate_acceleration(r)
    f_dot = np.hstack((x[3:], a))
    pbar.update(t)
    return f_dot

def dynamics_cart_w_STM(t, z, model, pbar): 
    N = int(1/2 * (np.sqrt(4*len(z) + 1) - 1))
    
    r = z[:3].reshape((1,-1))
    v = z[3:6].reshape((1,-1))
    a = model.generate_acceleration(r)
    dadx = model.generate_dadx(r)
    x_dot = np.hstack((v[0], a))
    
    zeros_3x3 = np.zeros((3,3))
    ident_3x3 = np.eye(3)
    dfdx = np.block([
        [zeros_3x3, ident_3x3],
        [dadx,      zeros_3x3]
        ])
    phi = z[N:].reshape((N,N))
    phi_dot = dfdx@phi
    
    z_dot = np.hstack((x_dot, phi_dot.reshape(-1)))
    pbar.update(t)
    return z_dot

def dynamics_OE(t, x, lpe, pbar): 
    OE = x
    # import tensorflow as tf
    dOEdt = lpe.dOE_dt(OE)
    if np.any(np.isnan(dOEdt)):
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)
        np.set_printoptions(formatter={'float': "{0:0.5e}".format})
        raise ValueError(f"NAN encountered in integration at time {t} and state \n {x}")
    pbar.update(t)
    return dOEdt

def dynamics_OE_w_STM(t, z, lpe, pbar): 
    N = int(1/2 * (np.sqrt(4*len(z) + 1) - 1))
    OE = z[:N]
    dOEdt = lpe.dOE_dt(OE)
    dOEdt_dOE = lpe.dOE_dt_dx(OE)
        
    phi = z[N:].reshape((N,N))
    phi_dot = dOEdt_dOE@phi
    
    z_dot = np.hstack((dOEdt, phi_dot.reshape(-1)))
    pbar.update(t)
    return z_dot