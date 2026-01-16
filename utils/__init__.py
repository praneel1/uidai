"""
Utility modules for Aadhaar data analysis.

This package provides functions for data loading, preprocessing,
feature engineering, visualization, and machine learning.
"""

from .data_loader import load_aadhaar_data, format_dates
from .preprocessing import merge_datasets, clean_merged_data, validate_data
from .features import (
    add_activity_columns,
    add_activity_type,
    add_temporal_features,
    add_ratio_features,
    add_all_features
)
from .visualizations import plot_state_map, plot_bar, plot_confusion_matrix
from .ml_models import (
    train_activity_predictor,
    train_activity_classifier,
    train_activity_clusters,
    train_all_models
)

__all__ = [
    'load_aadhaar_data',
    'format_dates',
    'merge_datasets',
    'clean_merged_data',
    'validate_data',
    'add_activity_columns',
    'add_activity_type',
    'add_temporal_features',
    'add_ratio_features',
    'add_all_features',
    'plot_state_map',
    'plot_bar',
    'plot_confusion_matrix',
    'train_activity_predictor',
    'train_activity_classifier',
    'train_activity_clusters',
    'train_all_models',
]
