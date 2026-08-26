"""EviSSL - Evidential semi-supervised skin lesion segmentation.

A compact research codebase that replaces confidence-threshold pseudo-labelling
with a principled evidential (Dirichlet) uncertainty signal, trained on a
~1M-parameter separable U-Net instead of the conventional 31M-parameter U-Net.

Public entry points:
    evissl.config.load_config   - YAML config loading with ``_base_`` inheritance
    evissl.models.build_model   - model registry
    evissl.engine               - supervised / semi-supervised trainers
    evissl.eval.evaluate        - metric + calibration + statistics suite
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
