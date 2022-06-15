from GravNN.Support.transformations import cart2sph, invert_projection, project_acceleration
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Layers import PreprocessingLayer, PostprocessingLayer

import numpy as np
import pandas as pd
import tensorflow as tf

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
        return a.squeeze()

    def generate_potential(self, X):
        X = np.array(X).reshape((-1,3))
        u = self.poly_model.compute_potential(positions=X)
        return u.squeeze()


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

        # configure preprocessing layers
        x_transformer = config['x_transformer'][0]
        u_transformer = config['u_transformer'][0]

        x_preprocessor = PreprocessingLayer(x_transformer.min_, x_transformer.scale_, tf.float64)
        u_postprocessor = PostprocessingLayer(u_transformer.min_, u_transformer.scale_, tf.float64)

        self.gravity_model.x_preprocessor = x_preprocessor
        self.gravity_model.u_postprocessor = u_postprocessor

    def generate_acceleration(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        a_model = self.gravity_model.generate_acceleration(R).numpy() # this is currently a_r > 0
        # a_model *= -1 # to make dynamics work, this will be multiplied by -1    
        
        if not self.removed_pm:
            return a_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            a_pm_sph = np.zeros((len(R), 3))
            a_pm_sph[:,0] = -self.planet.mu/r**2
            r_sph = cart2sph(R)
            a_pm_xyz = invert_projection(r_sph, a_pm_sph)
            return (a_pm_xyz + a_model).squeeze() # a_r < 0


    def generate_dadx(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        dadx = self.gravity_model.generate_dU_dxdx(R).numpy() # this is also > 0
        return dadx.squeeze()

    def generate_potential(self, X):
        R = np.array(X).reshape((-1,3)).astype(np.float32)
        U_model = self.gravity_model.generate_potential(R).numpy() # this is also > 0
        if not self.removed_pm:
            return U_model.squeeze()
        else:
            r = np.linalg.norm(R, axis=1)
            U_pm = np.zeros((len(R), 1))
            U_pm[:,0] = -self.planet.mu/r
            return (U_pm + U_model).squeeze()