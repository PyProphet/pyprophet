"""
This module provides the main tools for statistical scoring, error estimation, and hypothesis testing
in targeted proteomics and glycoproteomics workflows. It includes modules for semi-supervised
learning, feature scaling, classifier integration, and context-specific inference.

Submodules:
-----------
- `data_handling`: Utilities for handling and processing data, including feature scaling,
  ranking, and validation.
- `classifiers`: Implements various classifiers (e.g., LDA, SVM, XGBoost) for scoring.
- `semi_supervised`: Implements semi-supervised learning workflows for iterative scoring.
- `runner`: Defines workflows for running PyProphet, including learning and weight application.
- `pyprophet`: Core functionality for orchestrating scoring and error estimation workflows.

Dependencies:
-------------
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `loguru`
- `click`
"""
