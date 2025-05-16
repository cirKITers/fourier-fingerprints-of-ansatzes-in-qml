from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    visualize_coefficients_correlated,
    visualize_parameters_correlated,
    visualize_parameters_coefficients_correlated,
    visualize_coefficients_correlated_control,
    visualize_model,
    visualize_data,
    visualize_input_data,
    visualize_coefficients_decay,
    visualize_spectrum,
    visualize_coefficients_correlated_filtered_weighted,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_coefficients_correlated,
                inputs={
                    "df": "coefficients_correlated_filtered",
                    "model": "model",
                    "discard_negative": "params:training.positive_coeffs_only",
                    "triu": "params:coefficients.triu",
                },
                outputs="fig_coefficients_correlated",
                name="visualize_coefficients_correlated",
            ),
            node(
                func=visualize_coefficients_correlated_filtered_weighted,
                inputs={
                    "df": "coefficients_correlated_filtered_weighted",
                    "model": "model",
                    "discard_negative": "params:training.positive_coeffs_only",
                    "triu": "params:coefficients.triu",
                },
                outputs="fig_coefficients_correlated_weighted",
                name="visualize_coefficients_correlated_filtered_weighted",
            ),
            node(
                func=visualize_coefficients_decay,
                inputs={
                    "df": "coefficients_decay",
                    "model": "model",
                },
                outputs="fig_coefficients_decay",
                name="visualize_coefficients_decay",
            ),
            node(
                func=visualize_spectrum,
                inputs={
                    "df": "coefficients_filtered",
                    "model": "model",
                    "mts": "params:data.mts",
                    "mfs": "params:data.mfs",
                },
                outputs="fig_spectrum",
                name="visualize_spectrum",
            ),
            node(
                func=visualize_parameters_correlated,
                inputs={
                    "df": "coefficients_correlated",
                    "model": "model",
                    "triu": "params:coefficients.triu",
                },
                outputs="fig_parameters_correlated",
                name="visualize_parameters_correlated",
            ),
            node(
                func=visualize_parameters_coefficients_correlated,
                inputs={
                    "df": "coefficients_correlated",
                    "model": "model",
                },
                outputs="fig_parameters_coefficients_correlated",
                name="visualize_parameters_coefficients_correlated",
            ),
            # node(
            #     func=visualize_coefficients_correlated_control,
            #     inputs={
            #         "df": "coefficients_correlated_control",
            #         "model": "model",
            #     },
            #     outputs="fig_coefficients_correlated_control",
            #     name="visualize_coefficients_correlated_control",
            # ),
        ]
    )


def create_randcoeffs_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_coefficients_correlated,
                inputs={
                    "df": "random_coefficients_correlated",
                    "model": "model",
                    "discard_negative": "params:training.positive_coeffs_only",
                    "triu": "params:coefficients.triu",
                },
                outputs="fig_random_coefficients_correlated",
                name="visualize_random_coefficients_correlated",
            ),
        ]
    )


def create_model_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_model,
                inputs={
                    "model": "trained_model",
                    "data_loader": "train_loader",
                    "noise_params": "params:model.noise_params",
                },
                outputs={"model": "fig_model_trained"},
                name="visualize_model_trained",
            ),
            node(
                func=visualize_model,
                inputs={
                    "model": "model",
                    "data_loader": "valid_loader",
                    "noise_params": "params:model.noise_params",
                },
                outputs={"model": "fig_model_initial"},
                name="visualize_model_initial",
            ),
        ]
    )


def create_data_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                func=visualize_spectrum,
                inputs={
                    "df": "coeffs_target",
                    "model": "model",
                    "mts": "params:data.mts",
                    "mfs": "params:data.mfs",
                },
                outputs="fig_data_spectrum",
                name="visualize_spectrum",
            ),
            node(
                func=visualize_input_data,
                inputs={
                    "model": "trained_model",
                    "data_loader": "train_loader",
                    "scalers": "scalers",
                    "noise_params": "params:model.noise_params",
                },
                outputs="fig_input_data",
                name="visualize_model_train_data",
            ),
            # node(
            #     func=visualize_data,
            #     inputs={
            #         "model": "trained_model",
            #         "data_loader": "train_loader",
            #         "scalers": "scalers",
            #         "noise_params": "params:model.noise_params",
            #     },
            #     outputs={"model": "fig_model_train"},
            #     name="visualize_model_train_data",
            # ),
            # node(
            #     func=visualize_data,
            #     inputs={
            #         "model": "trained_model",
            #         "data_loader": "valid_loader",
            #         "scalers": "scalers",
            #         "noise_params": "params:model.noise_params",
            #     },
            #     outputs={"model": "fig_model_valid"},
            #     name="visualize_model_valid_data",
            # ),
        ]
    )
