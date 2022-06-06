from GravNN.Support.transformations import cart2sph, invert_projection, project_acceleration
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
import numpy as np
import pandas as pd

class ExtFloat:
    def __init__(self,value):
        self.value = value
    def numpy(self):
        return self.value

class simpleGravityModel:
    def __init__(self, mu):
        self.mu = mu

    def generate_acceleration(self, X):
        r = np.linalg.norm(X)
        a_r = self.mu/r**2
        a_theta = 0.0
        a_phi = 0.0
        X = np.array(X).reshape((-1,3))
        x_sph = cart2sph(X)
        a_sph = np.array([[a_r, a_theta, a_phi]]).reshape((1,3))
        a_cart = invert_projection(x_sph, a_sph)
        return ExtFloat(a_cart)

class noneGravityModel:
    def __init__(self, mu):
        self.mu = mu

    def generate_acceleration(self, X):
        return ExtFloat(np.array([[0,0,0]]))


class polyhedralGravityModel():
    def __init__(self, celestial_body, obj_file):
        self.planet = celestial_body
        self.poly_model = Polyhedral(celestial_body, obj_file)

    def generate_acceleration(self, X):
        X = np.array(X).reshape((-1,3))
        a = self.poly_model.compute_acceleration(positions=X, pbar=False)
        return a

    def generate_potential(self, X):
        X = np.array(X).reshape((-1,3))
        u = self.poly_model.compute_potential(positions=X)
        return u


class pinnGravityModel():
    def __init__(self, df_file):
        df = pd.read_pickle(df_file)
        config, gravity_model = load_config_and_model(df.iloc[-1]['id'], df)
        self.config = config
        self.gravity_model = gravity_model
        self.planet = config['planet'][0]
        removed_pm = config.get('remove_point_mass', [False])[0]
        deg_removed = config.get('deg_removed', [-1])[0]
        if removed_pm or deg_removed > -1:
            self.removed_pm = True
        else:
            self.removed_pm = False

    def generate_acceleration(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        a_model = self.gravity_model.generate_acceleration(R).numpy() # this is currently a_r > 0
        # a_model *= -1 # to make dynamics work, this will be multiplied by -1    
        
        if not self.removed_pm:
            return a_model
        else:
            r = np.linalg.norm(R, axis=1)
            a_pm_sph = np.zeros((len(R), 3))
            a_pm_sph[:,0] = -self.planet.mu/r**2
            r_sph = cart2sph(R)
            a_pm_xyz = invert_projection(r_sph, a_pm_sph)
            return a_pm_xyz + a_model # a_r < 0


    def generate_potential(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        U_model = self.gravity_model.generate_potential(R).numpy() # this is also > 0
        if not self.removed_pm:
            return U_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            U_pm = np.zeros((len(R), 1))
            U_pm = -self.planet.mu/r
            return (U_pm + U_model).squeeze()