"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from entangling_the_waves.pipelines.data_generation.pipeline import (
    create_pipeline as create_data_generation_pipeline,
)
from entangling_the_waves.pipelines.data_science.coefficients.pipeline import (
    create_pipeline as create_coefficients_pipeline,
)
from entangling_the_waves.pipelines.data_science.training.pipeline import (
    create_pipeline as create_training_pipeline,
)

from entangling_the_waves.pipelines.visualization.pipeline import (
    create_pipeline as create_visualization_pipeline,
)

from entangling_the_waves.pipelines.visualization.pipeline import (
    create_model_pipeline as create_model_visualization_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "__default__": create_data_generation_pipeline()
        + create_coefficients_pipeline()
        + create_training_pipeline()
        + create_visualization_pipeline()
        + create_model_visualization_pipeline(),
        "training": create_data_generation_pipeline()
        + create_training_pipeline()
        + create_model_visualization_pipeline(),
        "coefficients": create_data_generation_pipeline()
        + create_coefficients_pipeline()
        + create_visualization_pipeline(),
    }
    return pipelines
