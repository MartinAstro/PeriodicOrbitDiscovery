import os
from pprint import pprint

import GravNN
from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    directory = os.path.dirname(GravNN.__file__) + "/../"
    df_file = directory + "Data/Dataframes/eros_poly_061523.data"
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist": [100000],
        "N_train": [95000],
        "grav_file": [Eros().obj_200k],
        "N_val": [5000],
        "num_units": [40],
        "loss_fcns": [["percent"]],
        "jit_compile": [True],
        "lr_anneal": [False],
        "eager": [False],
        "learning_rate": [0.001],
        "batch_size": [2**16],
        "epochs": [10000],
        "preprocessing": [["pines", "r_inv"]],
        "PINN_constraint_fcn": ["pinn_a"],
        "gravity_data_fcn": [get_poly_data],
        "dropout": [0.0],
        "fuse_models": [True],
        "print_interval": [10],
        "radius_max": [Eros().radius * 10],
        "remove_point_mass": [False],
        # "dtype": ["float64"],
    }
    args = configure_run_args(config, hparams)
    configs = [run(*args[0])]
    save_training(df_file, configs)


def run(config):
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)
    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
