import numpy as np
import pandas as pd
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model


class ModelInterface:
    def __init__(self):
        pass

    def compute_acceleration(self, X):
        X = np.array(X).reshape((-1, 3))
        a = self.gravity_model.compute_acceleration(X)
        try:
            a = a.numpy()
        except Exception:
            pass
        return a.squeeze()

    def compute_potential(self, X):
        X = np.array(X).reshape((-1, 3))
        u = self.gravity_model.compute_potential(X)
        try:
            u = u.numpy()
        except Exception:
            pass
        return u.squeeze()


class ExtFloat:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class simpleGravityModel(ModelInterface):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.gravity_model = PointMass(mu)


class noneGravityModel:
    def __init__(self, mu):
        self.mu = mu

    def compute_acceleration(self, X):
        return ExtFloat(np.array([[0, 0, 0]]))


class polyhedralGravityModel(ModelInterface):
    def __init__(self, celestial_body, obj_file):
        super().__init__()
        self.planet = celestial_body
        self.gravity_model = Polyhedral(celestial_body, obj_file)


class pinnGravityModel(ModelInterface):
    def __init__(self, df_file):
        super().__init__()
        df = pd.read_pickle(df_file)
        config, gravity_model = load_config_and_model(df.iloc[-1]["id"], df)
        self.config = config
        self.gravity_model = gravity_model
        self.planet = config["planet"][0]
        self.dtype = np.float32 if self.config["dtype"][0] == "float32" else np.float64

    def compute_dadx(self, X):
        R = np.array(X).reshape((-1, 3)).astype(self.dtype)
        dadx = self.gravity_model.compute_dU_dxdx(R).numpy()  # this is also > 0
        return dadx.squeeze()

    def compute_disturbing_potential(self, X):
        R = np.array(X).reshape((-1, 3)).astype(self.dtype)
        U_model = self.gravity_model.compute_disturbing_potential(R).numpy()
        return U_model.squeeze()
