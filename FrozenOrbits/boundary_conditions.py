import numpy as np



def get_mirror_theorem_bc(y_guess, lpe, true_jac=False):
    R0 = y_guess[0:3,0]

    def bc(ya, yb, p=None):
        y_f = yb[1]
        x_dot_f = yb[3]

        bc_res = np.hstack([y_f, x_dot_f]) #1e4 is approx radius of asteroid
        return bc_res

    def bc_jac(ya, yb, p=None):
        n = len(ya)
        dbc_dya = np.zeros((2,3))
        dbc_dyb = np.zeros((2,3))

        gx = -(yb[0] - ya[0])/np.linalg.norm(yb-ya)
        gy = -(yb[1] - ya[1])/np.linalg.norm(yb-ya)
        gz = -(yb[2] - ya[2])/np.linalg.norm(yb-ya)

        dbc_dya[0] = [-1,-1,-1,0,0,0]
        dbc_dya[1] = [0,0,0,-1,-1,-1]
        dbc_dya[2] = [gx,gy,gz,0,0,0]
        dbc_dya[3] = [1,0,0,0,0,0]
        dbc_dya[4] = [0,1,0,0,0,0]
        dbc_dya[5] = [0,0,1,0,0,0]

        dbc_dyb = -dbc_dya
        dbc_dyb[3:] = 0.0

        return dbc_dya, dbc_dyb

    def fun(x,y,p=None):
        "Return the first-order system"
        R = y[0:3]
        V = y[3:6]
        a = lpe.model.generate_acceleration(R.T) 

        dxdt = np.vstack((V, a.T))
        return dxdt

    def fun_jac(x, y, p=None):
        n = len(y)
        m = len(y[0])
        df_dy = np.zeros((n,n,m))

        df_dy[0,:,:] = np.broadcast_to([0,0,0,1,0,0], (m,6)).T
        df_dy[1,:,:] = np.broadcast_to([0,0,0,0,1,0], (m,6)).T
        df_dy[2,:,:] = np.broadcast_to([0,0,0,0,0,1], (m,6)).T

        dU_dxdx = lpe.model.gravity_model.generate_dU_dxdx(y[:3].T).numpy() # this actually da_dx
        df_dy[3:6,0:3,:] = dU_dxdx.T
        return df_dy


    if true_jac:
        fun_jac = fun_jac
        bc_jac = bc_jac
    else:
        fun_jac = None
        bc_jac = None


    return fun, bc, fun_jac, bc_jac


def get_pos_bc_fcns(y_guess, lpe, true_jac=False):
    R0 = y_guess[0:3,0]

    def bc(ya, yb, p=None):

        xa, xb = ya[0:3], yb[0:3]
        va, vb = ya[3:6], yb[3:6]

        dX1 = ((xb[0] - xa[0]) + (xb[1] - xa[1]) + (xb[2] - xa[2])) 
        dX2 = ((vb[0] - va[0]) + (vb[1] - va[1]) + (vb[2] - va[2]))
        dX3 = np.linalg.norm(xb - xa) 

        dX4 = (xa[0] - R0[0]) 
        dX5 = (xa[1] - R0[1])
        dX6 = (xa[2] - R0[2]) 

        bc_res = np.hstack([dX1, dX2, dX3, dX4, dX5, dX6]) #1e4 is approx radius of asteroid
        return bc_res

    def bc_jac(ya, yb, p=None):
        n = len(ya)
        dbc_dya = np.zeros((n,n))
        dbc_dyb = np.zeros((n,n))

        gx = -(yb[0] - ya[0])/np.linalg.norm(yb-ya)
        gy = -(yb[1] - ya[1])/np.linalg.norm(yb-ya)
        gz = -(yb[2] - ya[2])/np.linalg.norm(yb-ya)

        dbc_dya[0] = [-1,-1,-1,0,0,0]
        dbc_dya[1] = [0,0,0,-1,-1,-1]
        dbc_dya[2] = [gx,gy,gz,0,0,0]
        dbc_dya[3] = [1,0,0,0,0,0]
        dbc_dya[4] = [0,1,0,0,0,0]
        dbc_dya[5] = [0,0,1,0,0,0]

        dbc_dyb = -dbc_dya
        dbc_dyb[3:] = 0.0

        return dbc_dya, dbc_dyb

    def fun(x,y,p=None):
        "Return the first-order system"
        R = y[0:3]
        V = y[3:6]
        a = lpe.model.generate_acceleration(R.T) 

        dxdt = np.vstack((V, a.T))
        return dxdt

    def fun_jac(x, y, p=None):
        n = len(y)
        m = len(y[0])
        df_dy = np.zeros((n,n,m))

        df_dy[0,:,:] = np.broadcast_to([0,0,0,1,0,0], (m,6)).T
        df_dy[1,:,:] = np.broadcast_to([0,0,0,0,1,0], (m,6)).T
        df_dy[2,:,:] = np.broadcast_to([0,0,0,0,0,1], (m,6)).T

        dU_dxdx = lpe.model.gravity_model.generate_dU_dxdx(y[:3].T).numpy() # this actually da_dx
        df_dy[3:6,0:3,:] = dU_dxdx.T
        return df_dy


    if true_jac:
        fun_jac = fun_jac
        bc_jac = bc_jac
    else:
        fun_jac = None
        bc_jac = None


    return fun, bc, fun_jac, bc_jac

def get_milankovitch_bc_fcns(y_guess, lpe, true_jac=False):
    OE_i = y_guess[:,0]

    def bc(ya, yb, p=None):

        h1_a, h2_a, h3_a, e1_a, e2_a, e3_a, l_a = ya
        h1_b, h2_b, h3_b, e1_b, e2_b, e3_b, l_b = yb

        ha = np.array([h1_a, h2_a, h3_a])
        hb = np.array([h1_b, h2_b, h3_b])
        ea = np.array([e1_a, e2_a, e3_a])
        eb = np.array([e1_b, e2_b, e3_b])

        h_mag = np.linalg.norm(ha)
        # # Angular Momentum
        # dX1 = (h1_b - h1_a) + (e1_b - e1_a) 
        # dX2 = (h2_b - h2_a) + (e2_b - e2_a)
        # dX3 = (h3_b - h3_a) + (e3_b - e3_a) 

        # dX4 = (h1_a - OE_i[0]) # 
        # dX5 = (h2_a - OE_i[1]) # e
        # dX6 = (h3_a - OE_i[2]) # 

        # dX7 = (l_a - l_b) #+ (l_a - l_b)/1E3 # This second term is required to avoid singularity in jacobian.


        # Angular Momentum
        dX1 = ((h1_b - h1_a) + (h2_b - h2_a) + (h3_b - h3_a)) / h_mag
        dX2 = (e1_b - e1_a) + (e2_b - e2_a) + (e3_b - e3_a) 

        # dX3 = np.linalg.norm(hb - ha) 
        # dX3 = l_b - l_a
        dX3 = (l_a - OE_i[6]) + (l_b - OE_i[6])/1E3
        dX4 = np.linalg.norm(eb - ea)

        dX5 = (h1_a - OE_i[0]) 
        dX6 = (h2_a - OE_i[1]) 
        dX7 = (h3_a - OE_i[2]) # fix angular momentum 
        # dX6 = np.linalg.norm(ea) - np.linalg.norm(OE_i[3:6]) # Fix the eccentricity magnitude
        # dX7 = (l_a - l_b) #+ (l_a - l_b)/1E3 # This second term is required to avoid singularity in jacobian.

        bc_res = np.hstack([dX1, dX2, dX3, dX4, dX5, dX6, dX7]) #1e4 is approx radius of asteroid
        return bc_res


    def fun(x,y,p=None):
        "Return the first-order system"

        dxdt = lpe(y.T)
        dxdt = np.hstack([[
            dxdt['dh1dt'],
            dxdt['dh2dt'],
            dxdt['dh3dt'],
            dxdt['de1dt'],
            dxdt['de2dt'],
            dxdt['de3dt'],
            dxdt['dLdt'][:,0]
        ]])
        return dxdt

    if true_jac:
        NotImplementedError()
    else:
        fun_jac = None
        bc_jac = None

    return fun, bc, fun_jac, bc_jac
