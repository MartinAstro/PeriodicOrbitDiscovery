from GravNN.Support.transformations import cart2sph, invert_projection
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
        X = np.array(X).reshape((1,3))
        x_sph = cart2sph(X)
        a_sph = np.array([[a_r, a_theta, a_phi]]).reshape((1,3))
        a_cart = invert_projection(x_sph, a_sph)
        return ExtFloat(a_cart)


class polyhedralGravityModel():
    def __init__(self, celestial_body, obj_file):
        self.poly_model = Polyhedral(celestial_body, obj_file)

    def generate_acceleration(self, X):
        X = np.array(X).reshape((1,3))
        a = self.poly_model.compute_acceleration(positions=X, pbar=False)
        return ExtFloat(a)


class pinnGravityModel():
    def __init__(self, df_file):
        df = pd.read_pickle(df_file)
        _, gravity_model = load_config_and_model(df.iloc[0]['id'], df)
        self.gravity_model = gravity_model

    def generate_acceleration(self, X):
        X = np.array(X).reshape((1,3)).astype(np.float32)
        a = self.gravity_model.generate_acceleration(X).numpy()
        return a

