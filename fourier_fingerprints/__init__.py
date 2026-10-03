"""Fourier fingerprints, datasets, models, and training utilities."""

import jax

# float32 breaks the realness check of Coefficients.get_spectrum and makes the
# numerically zero Fourier coefficients (which enter the unpruned FCC) noisier
jax.config.update("jax_enable_x64", True)

__version__ = "0.1"
