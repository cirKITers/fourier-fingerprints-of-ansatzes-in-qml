from entangling_the_waves.helpers.coefficients import Coefficients
from qml_essentials.model import Model

import pandas as pd
from typing import Dict

import logging

log = logging.getLogger(__name__)


def calculate_coefficients(model: Model, samples: int, seed: int, noise_params: Dict):

    entangling_capability = Coefficients.numerical(
        model=model,
        samples=samples,
        seed=seed,
        inputs=[0],
        noise_params=noise_params,
        cache=False,
    )

    log.info(f"Calculated entangling capability: {entangling_capability}")

    return entangling_capability


def correlate(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())
