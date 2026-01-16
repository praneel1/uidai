"""
Machine Learning utilities for Aadhaar analysis.

This module provides functions for training and evaluating ML models
with proper documentation and metrics.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler
from constants import ACTIVITY_NAMES


def train_activity_predictor(
    df: pd.DataFrame,
    features: list = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Linear Regression model to predict total activity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target
    features : list, optional
        Feature columns to use. Default: ['day', 'month', 'is_weekend']
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    dict
        Dictionary containing:
        - model: Trained model
        - metrics: Performance metrics (MSE, RMSE, R²)
        - feature_importance: Feature coefficients
        - X_test, y_test: Test data for further analysis
        
    Examples
    --------
    >>> results = train_activity_predictor(df)
    >>> print(f"R² Score: {results['metrics']['r2']:.4f}")
    """
    if features is None:
        features = ['day', 'month', 'is_weekend']
    
    print("=" * 60)
    print("LINEAR REGRESSION: Predicting Total Activity")
    print("=" * 60)
    
    # Prepare data
    X = df[features].copy()
    y = df['total_activity'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nDataset Split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples: {len(X_test):,}")
    print(f"  Features: {features}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15}")
    print("-" * 50)
    print(f"{'MSE':<20} {mse_train:>14,.2f} {mse_test:>14,.2f}")
    print(f"{'RMSE':<20} {np.sqrt(mse_train):>14,.2f} {np.sqrt(mse_test):>14,.2f}")
    print(f"{'R² Score':<20} {r2_train:>14.4f} {r2_test:>14.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFeature Importance (by coefficient magnitude):")
    print("-" * 50)
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:<15} {row['coefficient']:>12.2f}")
    
    print(f"\nIntercept: {model.intercept_:,.2f}")
    print("=" * 60)
    
    return {
        'model': model,
        'metrics': {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'rmse_train': np.sqrt(mse_train),
            'rmse_test': np.sqrt(mse_test),
            'r2_train': r2_train,
            'r2_test': r2_test
        },
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test
    }


def train_activity_classifier(
    df: pd.DataFrame,
    features: list = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train Logistic Regression model to classify activity type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target
    features : list, optional
        Feature columns. Default: ['demo_ratio', 'bio_ratio', 'enrol_ratio', 
                                   'day_of_week', 'month']
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    dict
        Dictionary containing:
        - model: Trained model
        - metrics: Performance metrics (accuracy, classification report)
        - X_test, y_test: Test data
        
    Examples
    --------
    >>> results = train_activity_classifier(df)
    >>> print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    """
    if features is None:
        features = ['demo_ratio', 'bio_ratio', 'enrol_ratio', 'day_of_week', 'month']
    
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: Classifying Activity Type")
    print("=" * 60)
    
    # Filter out rows with no activity
    df_active = df[df['activity_type'] != -1].copy()
    
    # Prepare data
    X = df_active[features].copy()
    y = df_active['activity_type'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nDataset Split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples: {len(X_test):,}")
    print(f"  Features: {features}")
    print(f"  Classes: {sorted(y.unique())}")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\n{'Metric':<20} {'Training':<15} {'Testing':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {acc_train:>14.4f} {acc_test:>14.4f}")
    
    # Classification report
    print(f"\nClassification Report (Test Set):")
    print("-" * 50)
    class_names = [ACTIVITY_NAMES[i] for i in sorted(y.unique())]
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nConfusion Matrix:")
    print("-" * 50)
    print(f"{'':>15} ", end="")
    for name in class_names:
        print(f"{name[:12]:>12} ", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:15]:>15} ", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12,} ", end="")
        print()
    
    print("=" * 60)
    
    return {
        'model': model,
        'metrics': {
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
            'classification_report': classification_report(
                y_test, y_pred_test, target_names=class_names, output_dict=True
            ),
            'confusion_matrix': cm
        },
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test
    }


def train_activity_clusters(
    df: pd.DataFrame,
    features: list = None,
    n_clusters: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train K-Means clustering model to group similar activity patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features
    features : list, optional
        Feature columns. Default: ['demo_ratio', 'bio_ratio', 'enrol_ratio',
                                   'is_weekend', 'month']
    n_clusters : int
        Number of clusters (default: 3)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    dict
        Dictionary containing:
        - model: Trained model
        - scaler: StandardScaler used for normalization
        - metrics: Performance metrics (silhouette score, inertia)
        - cluster_centers: Cluster centers in original scale
        - labels: Cluster assignments
        
    Examples
    --------
    >>> results = train_activity_clusters(df, n_clusters=4)
    >>> print(f"Silhouette Score: {results['metrics']['silhouette']:.4f}")
    """
    if features is None:
        features = ['demo_ratio', 'bio_ratio', 'enrol_ratio', 'is_weekend', 'month']
    
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING: Grouping Activity Patterns")
    print("=" * 60)
    
    # Prepare data
    X = df[features].copy()
    
    print(f"\nDataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {features}")
    print(f"  Number of clusters: {n_clusters}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    inertia = model.inertia_
    
    print(f"\nModel Performance:")
    print("-" * 50)
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Inertia: {inertia:,.2f}")
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster Sizes:")
    print("-" * 50)
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count:>8,} samples ({pct:>5.1f}%)")
    
    # Cluster centers (in original scale)
    centers_scaled = model.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    
    print(f"\nCluster Centers (Original Scale):")
    print("-" * 50)
    centers_df = pd.DataFrame(centers_original, columns=features)
    print(centers_df.to_string())
    
    print("=" * 60)
    
    return {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'silhouette': silhouette,
            'inertia': inertia,
            'cluster_sizes': dict(zip(unique, counts))
        },
        'cluster_centers': centers_df,
        'labels': labels,
        'X_scaled': X_scaled
    }


def train_all_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train all three ML models with proper evaluation.
    
    Convenience function that trains:
    1. Linear Regression for activity prediction
    2. Logistic Regression for activity classification
    3. K-Means for activity clustering
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with all required features
        
    Returns
    -------
    dict
        Dictionary with results from all three models
        
    Examples
    --------
    >>> results = train_all_models(df)
    >>> print(f"Regression R²: {results['regression']['metrics']['r2_test']:.4f}")
    >>> print(f"Classification Accuracy: {results['classification']['metrics']['accuracy_test']:.4f}")
    >>> print(f"Clustering Silhouette: {results['clustering']['metrics']['silhouette']:.4f}")
    """
    print("\n" + "🤖" * 30)
    print("TRAINING ALL ML MODELS")
    print("🤖" * 30)
    
    results = {}
    
    # 1. Regression
    results['regression'] = train_activity_predictor(df)
    
    # 2. Classification
    results['classification'] = train_activity_classifier(df)
    
    # 3. Clustering
    results['clustering'] = train_activity_clusters(df)
    
    print("\n" + "✅" * 30)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("✅" * 30)
    print("\nSummary:")
    print(f"  Regression R² (test): {results['regression']['metrics']['r2_test']:.4f}")
    print(f"  Classification Accuracy (test): {results['classification']['metrics']['accuracy_test']:.4f}")
    print(f"  Clustering Silhouette: {results['clustering']['metrics']['silhouette']:.4f}")
    print("=" * 60 + "\n")
    
    return results
