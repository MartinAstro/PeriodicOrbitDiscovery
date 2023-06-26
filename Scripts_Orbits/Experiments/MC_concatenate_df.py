import glob
import os

import pandas as pd

import FrozenOrbits


def main(type):
    df = pd.DataFrame(
        {
            "T_0": [],
            "T_0_sol": [],
            "OE_0": [],
            "OE_0_sol": [],
            "X_0": [],
            "X_0_sol": [],
            "dOE_0": [],
            "dOE_sol": [],
            "dX_0": [],
            "dX_sol": [],
            "result": [],
            "valid": [],
            "solver_key": [],
            "index": [],
        },
    )

    # gather all of the data files in the directory using glob
    directory = os.path.dirname(FrozenOrbits.__file__) + f"/Data/MC/*{type}"
    data_files = glob.glob(directory + "*.data")

    # iterate through each file and append it to the dataframe
    for file in data_files:
        data = pd.read_pickle(file)
        df_k = pd.DataFrame().from_dict(data).set_index("index")
        df = pd.concat([df, df_k], axis=0)

    pd.to_pickle(df, directory + f"orbit_solutions_{type}.data")


if __name__ == "__main__":
    main("trad")
    main("equi")
    main("mil")
    main("cart")
