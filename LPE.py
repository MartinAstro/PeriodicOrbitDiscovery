from coordinate_transforms import *
import tensorflow as tf
import numpy as np



class LPE():
    def __init__(self, model, config, mu, OE, element_set='traditional'):
        self.model = model
        self.config = config 
        self.trad_OE = OE
        self.mu = tf.constant(mu, dtype=tf.float64, name='mu')
        self.element_set =  element_set.lower()
        self.eval_fcn = self.get_eval_fcn()
        # self.init_OE = self.get_init_OE()
    
    def get_eval_fcn(self):
        if self.element_set == "traditional":
            return self.dOEdt
        elif self.element_set == "delaunay":
            return self.dDelaunay_dt
        elif self.element_set == "equinoctial":
            return self.dEquinoctial_dt
        elif self.element_set == "milankovitch":
            return self.dMilankovitch_dt

    def __call__(self,OE):
        return self.eval_fcn(OE)
    
    def dOEdt(self, OE_inputs):
        """Calculate the derivative of traditional OE

        Args:
            OE (np.array): [N, 6] a,e,i,omega, Omega, M

        Returns:
            dict: derivatives
        """
        OE = tf.Variable(OE_inputs.astype(np.float64), dtype=tf.float64, name='orbit_elements')
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE) 
            r, v = oe2cart_tf(OE, self.mu)
            u_pred = self.model.generate_potential(r)
        dUdOE = tape.gradient(u_pred, OE)

        a, e, i = OE_inputs[:,0:3].T # transpose necessary to assign
        b = tf.cond(e > 1, lambda : a*np.sqrt(e**2-1.) , lambda : a*np.sqrt(1.-e**2))
        n = np.sqrt(self.mu/a**3)
        
        dOEdt = {
            'dadt' : 2.0/(n*a) * dUdOE[:,5],
            'dedt' : -b/(n*a**3*e)*dUdOE[:,3] + b**2/(n*a**4*e)*dUdOE[:,5],
            'didt' : -1.0/(n*a*b*np.sin(i))*dUdOE[:,4] + np.cos(i)/(n*a*b*np.sin(i))*dUdOE[:,3],
            'domegadt' : -np.cos(i)/(n*a*b*np.sin(i))*dUdOE[:,2] + b/(n*a**3*e)*dUdOE[:,1],
            'dOmegadt' : 1.0/(n*a*b*np.sin(i))*dUdOE[:,2],
            'dMdt' : n - 2.0/(n*a)*dUdOE[:,0] - (1-e**2)/(n*a**2*e)*dUdOE[:,1] #https://www.sciencedirect.com/topics/engineering/orbital-element -- Orbital Mechanics and Formation Flying 
        }
       
        return dOEdt
    
    def dEquinoctial_dt(self, equi_OE):
        # [0]p, [1]f, [2]g, [3]L, [4]h, [5]k
        p,f,g,L,h,k = equi_OE[:, 0:6].T
        equi_OE = tf.Variable(equi_OE.astype(np.float32), dtype=tf.float32, name='orbit_elements')
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(equi_OE) 
            OE = equinoctial2oe_tf(equi_OE)
            r, v = oe2cart_tf(OE, self.mu)
            u_pred = self.model.generate_potential(r)
        dUdOE = tape.gradient(u_pred, equi_OE)

        s = np.sqrt(1. + h**2 + k**2)
        w = 1. + f*np.cos(L) + g*np.sin(L)
        mu = self.mu
        # p, f, g, L, h, k
        dOEdt = {
            'dpdt' : 2.0*np.sqrt(p/mu) * (-g*dUdOE[:,1] + f*dUdOE[:,2] + dUdOE[:,3]),
            'dfdt' : 1.0/np.sqrt(mu*p)*(2.*p*g*dUdOE[:,0] - (1. - f**2 - g**2)*dUdOE[:,2] - g*s**2/2.0*(h*dUdOE[:,4] + k*dUdOE[:,5]) + (f+(1.0+w)*np.cos(L))*dUdOE[:,3]),
            'dgdt' : 1.0/np.sqrt(mu*p)*(-2.*p*f*dUdOE[:,0] + (1. - f**2 - g**2)*dUdOE[:,1] + f*s**2/2.0*(h*dUdOE[:,4] + k*dUdOE[:,5]) + (g+(1.0+w)*np.sin(L))*dUdOE[:,3]),
            'dLdt' : np.sqrt(mu*p)*(w/p)**2 + s**2/(2.0*np.sqrt(mu*p))*(h*dUdOE[:,4] + k*dUdOE[:,5]),
            'dhdt' : s**2/(2.0*np.sqrt(mu*p))*(h*(g*dUdOE[:,1] - f*dUdOE[:,2] - dUdOE[:,3]) - s**2/2.0*dUdOE[:,5]),
            'dkdt' : s**2/(2.0*np.sqrt(mu*p))*(k*(g*dUdOE[:,1] - f*dUdOE[:,2] - dUdOE[:,3]) + s**2/2.0*dUdOE[:,4])
        }

        return dOEdt

    def dDelaunay_dt(self, delaunay_OE):
        l, g, h, L, G, H= delaunay_OE[:, 0:6].T
        delaunay_OE = tf.Variable(delaunay_OE.astype(np.float32), dtype=tf.float32, name='orbit_elements')
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(delaunay_OE) 
            OE = delaunay2oe_tf(delaunay_OE, self.mu)
            r, v = oe2cart_tf(OE, self.mu)
            u_pred = self.model.generate_potential(r)
            H = tf.square(self.mu)/(2.0*tf.square(L.astype(np.float32))) + u_pred
        dUdOE = tape.gradient(H, delaunay_OE)

        dOEdt = {
            'dldt' : -dUdOE[:,3],
            'dgdt' : -dUdOE[:,4],
            'dhdt' : -dUdOE[:,5],
            'dLdt' : -dUdOE[:,0],
            'dGdt' : -dUdOE[:,2],
            'dHdt' : -dUdOE[:,3]
        }
       
        return dOEdt

    def dMilankovitch_dt(self, mil_OE):
        milankovitch_OE = tf.Variable(mil_OE.astype(np.float32), dtype=tf.float32, name='orbit_elements')
        H = milankovitch_OE[:, 0:3]
        e = milankovitch_OE[:, 3:6]
        L = milankovitch_OE[:, 6]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(milankovitch_OE) 
            r,v = milankovitch2cart_tf(milankovitch_OE, self.mu)
            u_pred = self.model.generate_potential(r)
        dUdOE = tape.gradient(u_pred, milankovitch_OE)

        dUdH = dUdOE[0,0:3].numpy()
        dUde = dUdOE[0,3:6].numpy()
        dUdl = dUdOE[0,6].numpy()

        H_mag = np.linalg.norm(H)
        e_mag = np.linalg.norm(e)
        r_mag = np.linalg.norm(r)

        z_hat = np.array([0.0,0.0,1.0])
        rVec = r[0,:]
        
        HVec = H[0,:]#.numpy()
        eVec = e[0,:]#.numpy()
        LVal = L[0]#.numpy()
        
        r_hat = rVec/r_mag

        term1 = (HVec + H_mag*z_hat)/(H_mag + np.dot(z_hat, HVec))#
        term2 = (1.0-e_mag**2)/H_mag**2
        term3 = (2.0 + np.dot(r_hat, eVec))*r_hat + eVec
        term4 = np.dot(z_hat, eVec)/(H_mag*(H_mag + np.dot(z_hat, HVec))) 

        H_dt = np.cross(HVec, dUdH) + np.cross(eVec, dUde) + term1*dUdl # good
        e_dt = np.cross(eVec, dUdH) + term2*np.cross(HVec,dUde) + 1.0/H_mag*(term3 - term4*HVec)*dUdl
        L_dt = -np.dot(term1,dUdH) - 1.0/H_mag*np.dot((term3 - term4*HVec), dUde) + H_mag/r_mag**2 

        dOEdt = {
            'dh1dt' : H_dt[0],
            'dh2dt' : H_dt[1],
            'dh3dt' : H_dt[2],
            'de1dt' : e_dt[0],
            'de2dt' : e_dt[1],
            'de3dt' : e_dt[2],
            'dLdt' : tf.constant(L_dt, dtype=H_dt[0].dtype)
        }
       
        for val in dOEdt.values():
            if tf.math.is_nan(val).numpy():
                pass


        return dOEdt