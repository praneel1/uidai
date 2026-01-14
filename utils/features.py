"""
Feature engineering utilities for Aadhaar analysis.

This module handles creation of derived features and transformations.
"""
import pandas as pd
import numpy as np
from constants import (
    LAKH,
    ACTIVITY_ENROLMENT,
    ACTIVITY_BIOMETRIC,
    ACTIVITY_DEMOGRAPHIC,
    ACTIVITY_NONE,
    ACTIVITY_COLUMN_MAP,
    WEEKEND_DAYS
)


def add_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add total activity columns by summing age groups.
    
    Creates:
    - total_demo_updates: Sum of demographic updates
    - total_bio_updates: Sum of biometric updates
    - total_enrolments: Sum of enrolments
    - total_activity: Sum of all activities
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with age-group columns
        
    Returns
    -------
    pd.DataFrame
        Dataset with added activity columns
    """
    print("Adding activity columns...")
    
    df["total_demo_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]
    df["total_bio_updates"] = df["bio_age_5_17"] + df["bio_age_17_"]
    df["total_enrolments"] = (
        df["age_0_5"] + 
        df["age_5_17"] + 
        df["age_18_greater"]
    )
    df["total_activity"] = (
        df["total_demo_updates"] + 
        df["total_bio_updates"] + 
        df["total_enrolments"]
    )
    
    print("✓ Added 4 activity columns")
    return df


def add_activity_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine dominant activity type for each row.
    
    Activity types:
    - 0: Enrolment dominant
    - 1: Biometric update dominant
    - 2: Demographic update dominant
    - -1: No activity
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with activity columns
        
    Returns
    -------
    pd.DataFrame
        Dataset with activity_type column
    """
    print("Determining activity types...")
    
    # Find column with maximum value
    df["activity_type"] = df[[
        "total_demo_updates", 
        "total_bio_updates", 
        "total_enrolments"
    ]].idxmax(axis=1)
    
    # Map column names to numeric codes
    df["activity_type"] = df["activity_type"].map(ACTIVITY_COLUMN_MAP)
    
    # Mark rows with no activity
    no_activity_mask = (
        (df["total_enrolments"] == 0) & 
        (df["total_bio_updates"] == 0) & 
        (df["total_demo_updates"] == 0)
    )
    df.loc[no_activity_mask, "activity_type"] = ACTIVITY_NONE
    
    print("✓ Added activity_type column")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date-based features.
    
    Creates:
    - day: Day of month (1-31)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - month: Month (1-12)
    - is_weekend: 1 if Saturday/Sunday, 0 otherwise
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with date column
        
    Returns
    -------
    pd.DataFrame
        Dataset with temporal features
    """
    print("Adding temporal features...")
    
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    
    print("✓ Added 5 temporal features")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add activity ratio features.
    
    Creates:
    - demo_ratio: Proportion of demographic updates
    - bio_ratio: Proportion of biometric updates
    - enrol_ratio: Proportion of enrolments
    - is_inactive_day: 1 if no activity, 0 otherwise
    
    Uses np.where to avoid division by zero.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with activity columns
        
    Returns
    -------
    pd.DataFrame
        Dataset with ratio features
    """
    print("Adding ratio features...")
    
    # Calculate ratios, avoiding division by zero
    df['demo_ratio'] = np.where(
        df['total_activity'] > 0,
        df['total_demo_updates'] / df['total_activity'],
        0
    )
    
    df['bio_ratio'] = np.where(
        df['total_activity'] > 0,
        df['total_bio_updates'] / df['total_activity'],
        0
    )
    
    df['enrol_ratio'] = np.where(
        df['total_activity'] > 0,
        df['total_enrolments'] / df['total_activity'],
        0
    )
    
    df['is_inactive_day'] = (df['total_activity'] == 0).astype(int)
    
    print("✓ Added 4 ratio features")
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all feature engineering transformations.
    
    This is a convenience function that applies all feature
    engineering steps in the correct order.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned merged dataset
        
    Returns
    -------
    pd.DataFrame
        Dataset with all engineered features
        
    Examples
    --------
    >>> df = add_all_features(df)
    >>> print(df.columns)
    """
    df = add_activity_columns(df)
    df = add_activity_type(df)
    df = add_temporal_features(df)
    df = add_ratio_features(df)
    
    print("✓ All features added successfully")
    return df
