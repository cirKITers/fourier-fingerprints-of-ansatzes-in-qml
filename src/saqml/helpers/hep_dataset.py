import torch
from torch.utils.data import TensorDataset, DataLoader

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    QuantileTransformer,
    FunctionTransformer,
    Normalizer,
)
from sklearn.model_selection import train_test_split
import vector

import warnings
import copy

import logging

log = logging.getLogger(__name__)


def h5py_to_DataFrame(data, collection, n_events=None):
    """
    Converts a specified collection from an HDF5 (h5py) dataset into a pandas DataFrame.

    Code by @Lucas-vdH

    Parameters:
    -----------
    data : h5py.File
        The opened HDF5 file containing the dataset.

    collection : str
        The collection to extract from the file. Must be either 'partons' or 'jets'.

    Returns:
    --------
    df : pandas.DataFrame
        A DataFrame containing the reshaped and labeled data, indexed by EventID.
    """
    assert (
        collection == "partons" or collection == "jets"
    ), f"Unidentified collection of data. Expected 'partons' or 'jets', found '{collection}'"

    data = np.array(data[collection][:])

    # Reshape the array to 2D (axis_0 * axis_1, axis_2)
    reshaped_data = data.reshape(-1, data.shape[-1])

    df = pd.DataFrame(reshaped_data)

    if collection == "partons":
        df.columns = ["E", "px", "py", "pz", "particle_id", "charge"]
        df.drop(columns=["px", "py"], inplace=True)  # px and py are always 0
        df.insert(
            0, "EventID", np.floor_divide(df.index, 2)
        )  # Each two rows represent the same event

        # df['pz'] = df['pz'].abs() # pz of second parton is always negative
        df["particle_id"] = df["particle_id"].astype(int)
        df["charge"] = df["charge"].round(2)

        df["parton"] = df.groupby("EventID").cumcount() + 1  # Tag parton 1 and 2
        df = df.pivot(index="EventID", columns="parton")

        df.columns = [f"{col}_{p}" for col, p in df.columns]  # Flatten MultiIndex
        df.reset_index(inplace=True)
        df.set_index("EventID", inplace=True)

    elif collection == "jets":
        df.columns = ["E", "px", "py", "pz"]
        df.insert(
            0, "EventID", np.floor_divide(df.index, 5)
        )  # Each five rows represent the same event

        df["jets"] = df.groupby("EventID").cumcount() + 1  # Tag jet 1, 2, 3, 4 and 5
        df = df.pivot(index="EventID", columns="jets")

        df.columns = [f"{col}_{j}" for col, j in df.columns]  # Flatten MultiIndex
        df.reset_index(inplace=True)
        df.set_index("EventID", inplace=True)

    return df[:n_events]


def data_preprocessing(df, collection, encoded=False, scaling_method=None):
    """
    Preprocesses a DataFrame by applying feature engineering and optional scaling and encoding.

    Code by @Lucas-vdH

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing either jet-level or parton-level data.

    collection : str
        Indicates the type of data. Must be either 'partons' or 'jets'.

    encoded : bool, optional (default=False)
        If True and `collection` is 'partons', one-hot encodes 'particle_id' and 'charge' columns.

    scaling_method : str or None, optional (default=None)
        Scaling method to apply to selected numeric columns. Supported methods:
        'MinMax', 'Standard', 'MaxAbs', 'QuantileTransformer_Uniform',
        'QuantileTransformer_Normal', 'log', or None.

    Returns:
    --------
    df : pandas.DataFrame
        The processed DataFrame with engineered features and optional scaling.
    """
    # Checking the arguments are as expected
    assert (
        collection == "partons" or collection == "jets"
    ), f"Unidentified collection of data. Expected 'partons' or 'jets', found '{collection}'"
    scaling_methods = [
        "MinMax",
        "MinMaxPi",
        "MinMaxPiLog",
        "MinMaxZPi",
        "MinMaxZHalf",
        "MinMaxZ",
        "Standard",
        "MaxAbs",
        "QuantileTransformer_Uniform",
        "QuantileTransformer_Normal",
        "log",
        None,
    ]
    if scaling_method not in scaling_methods:
        raise ValueError(
            f"Scaling method not recognized, received {scaling_method}.\n"
            + f"Available methods are {scaling_methods}\n"
            + "No scaling was performed"
        )

    # Setting the scaler
    if scaling_method == "MinMax":
        scaler = MinMaxScaler()
    elif scaling_method == "MinMaxPi":
        mm_scaler = MinMaxScaler()
        scaler = FunctionTransformer(
            func=lambda x: 2 * np.pi * mm_scaler.fit_transform(x),
            inverse_func=lambda x: mm_scaler.inverse_transform((x / (2 * np.pi))),
            validate=True,
        )
    elif scaling_method == "MinMaxZPi":
        mm_scaler = MinMaxScaler()
        scaler = FunctionTransformer(
            func=lambda x: 2 * np.pi * mm_scaler.fit_transform(x) - np.pi,
            inverse_func=lambda x: mm_scaler.inverse_transform(
                ((x + np.pi) / (2 * np.pi))
            ),
            validate=True,
        )
    elif scaling_method == "MinMaxPiLog":
        mm_scaler = MinMaxScaler()
        scaler = FunctionTransformer(
            func=lambda x: 2 * np.pi * mm_scaler.fit_transform(np.log10(x + 1e-6)),
            inverse_func=lambda x: np.power(
                10, mm_scaler.inverse_transform((x / (2 * np.pi)))
            )
            - 1e-6,
            validate=True,
        )
    elif scaling_method == "MinMaxZ":
        mm_scaler = MinMaxScaler()
        scaler = FunctionTransformer(
            func=lambda x: mm_scaler.fit_transform(x) - 0.5,
            inverse_func=lambda x: mm_scaler.inverse_transform((x + 0.5)),
            validate=True,
        )
    elif scaling_method == "MinMaxZHalf":
        mm_scaler = MinMaxScaler()
        scaler = FunctionTransformer(
            func=lambda x: mm_scaler.fit_transform(x) - 0.5,
            inverse_func=lambda x: mm_scaler.inverse_transform((x + 0.5)),
            validate=True,
        )
    elif scaling_method == "Standard":
        scaler = StandardScaler()
    elif scaling_method == "MaxAbs":
        scaler = MaxAbsScaler()
    elif scaling_method == "QuantileTransformer_Uniform":
        scaler = QuantileTransformer(output_distribution="uniform")
    elif scaling_method == "QuantileTransformer_Normal":
        scaler = QuantileTransformer(output_distribution="normal")
    elif scaling_method == "log":
        scaler = FunctionTransformer(
            func=lambda x: np.log(x + 1e-6),
            inverse_func=lambda x: np.exp(x) - 1e-6,
            validate=True,
        )
    else:
        log.warning("No scaling was performed")
        scaler = False

    if collection == "jets":

        # Compute pt per jet
        jet_indices = [1, 2, 3, 4, 5]
        for i in range(1, 6):
            lorentz_vec = vector.array(
                {
                    "px": df[f"px_{i}"],
                    "py": df[f"py_{i}"],
                    "pz": df[f"pz_{i}"],
                    "E": df[f"E_{i}"],
                }
            )
            df[f"pt_{i}"] = lorentz_vec.pt

        # Jet multiplicity: number of non-zero-pt jets
        pt_cols = [f"pt_{i}" for i in jet_indices]
        df["n_jets"] = df[pt_cols].gt(0).sum(axis=1)

        # Get sorted pt values per row (descending)
        sorted_pts = df[pt_cols].apply(lambda row: sorted(row, reverse=True), axis=1)

        # Assign leading/subleading pt and their difference
        df["leading_pt"] = sorted_pts.apply(lambda x: x[0] if len(x) > 0 else 0)
        df["subleading_pt"] = sorted_pts.apply(lambda x: x[1] if len(x) > 1 else 0)
        df["pt_difference"] = df["leading_pt"] - df["subleading_pt"]

        # Scaling Energy, px, py, pz, pt, leading_pt, subleading_pt, pt_difference for every jet
        if scaler:
            scaler_ = copy.copy(scaler)
            for i in range(1, 6):
                for pj_ in ["px_", "py_", "pz_"]:
                    df[pj_ + str(i)] = df[
                        pj_ + str(i)
                    ].abs()  # abs to avoid log scaling issues

            # Collect all columns to scale
            cols_to_scale = [
                col
                for col in df.columns
                if col.startswith(("E_", "px_", "py_", "pz_", "pt_", "subleading"))
            ]

            df[cols_to_scale] = scaler_.fit_transform(df[cols_to_scale])
            df[["leading_pt"]] = scaler.fit_transform(df[["leading_pt"]])

    elif collection == "partons":

        # Create the 4-vectors
        zeros = np.zeros(len(df.index))
        p1 = vector.array({"px": zeros, "py": zeros, "pz": df["pz_1"], "E": df["E_1"]})

        p2 = vector.array({"px": zeros, "py": zeros, "pz": df["pz_2"], "E": df["E_2"]})

        total = p1 + p2
        diff = p1 - p2

        # Compute desired features
        df["E_total"] = total.E
        df["delta_E"] = df["E_1"] - df["E_2"]
        df["E_ratio"] = df["E_1"] / (df["E_2"] + 1e-6)
        df["pz_total"] = total.pz
        df["delta_pz"] = df["pz_1"] - df["pz_2"]
        df["M2"] = total.mass2
        df["E_CM"] = total.mass
        df["E^3"] = total.E**3
        df["E^4"] = total.E**4
        # df['rapidity'] = total.rapidity
        df["quark"] = df[["particle_id_1", "particle_id_2"]].min(axis=1)
        df["charge_total"] = df["charge_1"] + df["charge_2"]

        # # Add collision type feature based on which particles are colliding
        # conditions = [
        #     (df['particle_id_1'] == 21) & (df['particle_id_2'] == 21),
        #     ((df['particle_id_1'] == 21) & (df['particle_id_2'] != 21)) |
        #     ((df['particle_id_1'] != 21) & (df['particle_id_2'] == 21)),
        #     (df['particle_id_1'] != 21) & (df['particle_id_2'] != 21),
        # ]

        # choices = ['gluon-gluon', 'gluon-quark', 'quark-quark']

        # df['collision_type'] = np.select(conditions, choices)

        df["quark"] = df[["particle_id_1", "particle_id_2"]].min(axis=1)

        # One hot encoding particle_id and charge
        if encoded:
            df = pd.get_dummies(
                df,
                columns=[
                    "particle_id_1",
                    "particle_id_2",
                    "charge_1",
                    "charge_2",
                    "quark",
                ],
                dtype=float,
            )

        # Scaling Energy and pz
        if scaler:
            df["delta_E"] = df["delta_E"].abs()
            df["pz_2"] = df["pz_2"].abs()  # abs to avoid log scaling issues
            df["pz_total"] = df[
                "pz_total"
            ].abs()  # LHC detector is symmetric on z axis, so the sign is irrelevant

            # Collect all columns to scale
            cols_to_scale = [
                col
                for col in df.columns
                if col.startswith(("E_", "pz_", "delta_")) or col in ["M2", "rapidity"]
            ]

            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df, scaler


def get_loaders(
    batch_size,
    n_events,
    features,
    scaling_methods,
    labels,
    seed,
):
    """
    Prepares data loaders for training, validation, and testing datasets
    from HDF5 files containing partons and jets data.

    Code by @Lucas-vdH

    Parameters:
    -----------
    batch_size : int
        The number of samples per batch to load.

    n_events : int
        Number of events to load from the dataset.

    features : list of str, optional
        List of feature names to be extracted from the partons dataset.

    labels : list of str, optional
        List of label names to be extracted from the jets dataset.

    scaling_methods : list of str, optional
        List of scaling methods to be applied to partons and jets datasets.

    Returns:
    --------
    train_loader : DataLoader
        DataLoader for the training dataset.

    valid_loader : DataLoader
        DataLoader for the validation dataset.

    test_loader : DataLoader
        DataLoader for the test dataset.

    parton_scaler : Scaler
        Scaler object used for scaling partons data.

    jet_scaler : Scaler
        Scaler object used for scaling jets data.
    """
    # test_data_file = "data/pp-z-to-jets-500K-54167.h5"
    train_data_file = "data/pp-z-to-jets-500K-57246.h5"

    # test_data = h5py.File(test_data_file, "r")
    train_data = h5py.File(train_data_file, "r")

    partons_df, parton_scaler = data_preprocessing(
        h5py_to_DataFrame(train_data, "partons", n_events),
        "partons",
        encoded=False,
        scaling_method=scaling_methods[0],
    )
    jets_df, jet_scaler = data_preprocessing(
        h5py_to_DataFrame(train_data, "jets", n_events),
        "jets",
        scaling_method=scaling_methods[1],
    )

    # Dropping events where there are no recorded jets (energy/pt too small for detection)
    mask = jets_df["n_jets"] >= 1
    jets_df = jets_df[mask]
    partons_df = partons_df[mask]

    partons_df = partons_df[features]
    jets_df = jets_df[labels]

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    assert train_ratio + validation_ratio + test_ratio == 1

    # Split into train, validation and test sets
    partons_train, partons_test, jets_train, jets_test = train_test_split(
        partons_df, jets_df, test_size=1 - train_ratio
    )
    partons_valid, partons_test, jets_valid, jets_test = train_test_split(
        partons_test, jets_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    # Convert DataFrames to tensors
    partons_train_tensor = torch.tensor(partons_train.values, dtype=torch.float32)
    jets_train_tensor = torch.tensor(jets_train.values, dtype=torch.float32)

    partons_valid_tensor = torch.tensor(partons_valid.values, dtype=torch.float32)
    jets_valid_tensor = torch.tensor(jets_valid.values, dtype=torch.float32)

    partons_test_tensor = torch.tensor(partons_test.values, dtype=torch.float32)
    jets_test_tensor = torch.tensor(jets_test.values, dtype=torch.float32)

    # Create datasets pairing inputs (partons) and targets (jets)
    train_dataset = TensorDataset(partons_train_tensor, jets_train_tensor)
    valid_dataset = TensorDataset(partons_valid_tensor, jets_valid_tensor)
    test_dataset = TensorDataset(partons_test_tensor, jets_test_tensor)

    if batch_size < 1:
        batch_size = len(train_dataset)
    # Loaders
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(
        f"Length of the training dataset: {len(train_dataset)}\n"
        + f"Length of the validation dataset: {len(valid_dataset)}\n"
        + f"Length of the test dataset: {len(test_dataset)}"
    )
    print(f"Features: {features}\n" + f"Labels: {labels}")

    return train_loader, valid_loader, test_loader, parton_scaler, jet_scaler


def get_data():
    test_data_file = "data/pp-z-to-jets-500K-54167.h5"
    train_data_file = "data/pp-z-to-jets-500K-57246.h5"

    test_data = h5py.File(test_data_file, "r")
    train_data = h5py.File(train_data_file, "r")

    # Getting the data into DataFrames and preprocessing them
    # Note that the data exploration and visualization code blocks below will throw erros if encoded=True
    train_partons_df, train_parton_scaler = data_preprocessing(
        h5py_to_DataFrame(train_data, "partons"),
        "partons",
        encoded=False,
        scaling_method="log",
    )
    train_jets_df, train_jet_scaler = data_preprocessing(
        h5py_to_DataFrame(train_data, "jets"),
        "jets",
        encoded=False,
        scaling_method="log",
    )
    test_partons_df, test_parton_scaler = data_preprocessing(
        h5py_to_DataFrame(test_data, "partons"),
        "partons",
        encoded=False,
        scaling_method=None,
    )

    # Dropping events where there are no recorded jets (energy/pt too small for detection)
    mask = train_jets_df["n_jets"] >= 1
    train_jets_df = train_jets_df[mask]
    train_partons_df = train_partons_df[mask]

    return train_partons_df, train_jets_df, test_partons_df
