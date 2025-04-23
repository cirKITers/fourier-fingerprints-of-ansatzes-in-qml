"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from saqml.pipelines.data_generation.pipeline import (
    create_model_pipeline,
    create_fourier_pipeline,
    create_hep_pipeline,
)
from saqml.pipelines.data_science.coefficients.pipeline import (
    create_pipeline as create_coefficients_pipeline,
)
from saqml.pipelines.data_science.coefficients.pipeline import (
    create_randcoeffs_pipeline as create_randcoeffs_pipeline,
)
from saqml.pipelines.data_science.expressibility.pipeline import (
    create_pipeline as create_expressibility_pipeline,
)
from saqml.pipelines.data_science.training.pipeline import (
    create_pipeline as create_training_pipeline,
)

from saqml.pipelines.visualization.pipeline import (
    create_pipeline as create_visualization_pipeline,
)

from saqml.pipelines.visualization.pipeline import (
    create_model_pipeline as create_model_visualization_pipeline,
)
from saqml.pipelines.visualization.pipeline import (
    create_data_pipeline as create_data_visualization_pipeline,
)

from saqml.pipelines.visualization.pipeline import (
    create_randcoeffs_pipeline as create_randcoeffs_viz_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "__default__": create_model_pipeline()
        + create_fourier_pipeline()
        + create_coefficients_pipeline()
        + create_training_pipeline()
        + create_visualization_pipeline()
        + create_model_visualization_pipeline(),
        "visualize": create_model_pipeline(),
        "training_fourier": create_model_pipeline()
        + create_fourier_pipeline()
        + create_training_pipeline()
        + create_model_visualization_pipeline(),
        "training_hep": create_model_pipeline()
        + create_hep_pipeline()
        + create_training_pipeline()
        + create_data_visualization_pipeline(),
        "coefficients": create_model_pipeline()
        + create_coefficients_pipeline()
        + create_visualization_pipeline(),
        "randcoeffs": create_model_pipeline()
        + create_randcoeffs_pipeline()
        + create_randcoeffs_viz_pipeline(),
        "expressibility": create_model_pipeline() + create_expressibility_pipeline(),
        "coeffexpr": create_model_pipeline()
        + create_coefficients_pipeline()
        + create_expressibility_pipeline()
        + create_visualization_pipeline(),
    }
    return pipelines
