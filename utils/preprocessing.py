"""
Data preprocessing utilities for Aadhaar analysis.

This module handles data merging, cleaning, and validation.
"""
import pandas as pd
from typing import List
from constants import KEY_COLUMNS, NUMERIC_COLUMNS


def merge_datasets(
    demo: pd.DataFrame, 
    bio: pd.DataFrame, 
    enroll: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge three Aadhaar datasets on key columns.
    
    This function groups each dataset by key columns (date, state, district, pincode)
    and then performs outer joins to combine all data.
    
    Parameters
    ----------
    demo : pd.DataFrame
        Demographic updates data with columns: demo_age_5_17, demo_age_17_
    bio : pd.DataFrame
        Biometric updates data with columns: bio_age_5_17, bio_age_17_
    enroll : pd.DataFrame
        Enrolment data with columns: age_0_5, age_5_17, age_18_greater
        
    Returns
    -------
    pd.DataFrame
        Merged dataset with all columns from three sources
        
    Examples
    --------
    >>> merged = merge_datasets(demo, bio, enroll)
    >>> print(merged.columns)
    """
    print("Merging datasets...")
    
    # Group and aggregate each dataset
    demo_group = demo.groupby(KEY_COLUMNS, as_index=False)[
        ["demo_age_5_17", "demo_age_17_"]
    ].sum()
    
    bio_group = bio.groupby(KEY_COLUMNS, as_index=False)[
        ["bio_age_5_17", "bio_age_17_"]
    ].sum()
    
    enroll_group = enroll.groupby(KEY_COLUMNS, as_index=False)[
        ["age_0_5", "age_5_17", "age_18_greater"]
    ].sum()
    
    # Perform outer joins
    df = demo_group.merge(bio_group, on=KEY_COLUMNS, how='outer')
    df = df.merge(enroll_group, on=KEY_COLUMNS, how='outer')
    
    print(f"✓ Merged data: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def clean_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values and convert to appropriate types.
    
    Outer joins introduce NaN values where one action occurred but another
    did not. These are replaced with 0 and converted to integers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset with potential NaN values
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset with no NaN values in numeric columns
        
    Examples
    --------
    >>> df = clean_merged_data(df)
    >>> print(df[NUMERIC_COLUMNS].isna().sum().sum())
    0
    """
    print("Cleaning merged data...")
    
    # Fill NaN with 0 and convert to int
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].fillna(0).astype(int)
    
    print(f"✓ Cleaned {len(NUMERIC_COLUMNS)} numeric columns")
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate merged data integrity.
    
    Checks:
    1. Date column has no null values
    2. District column has no null values
    3. Pincode column has no null values
    4. No duplicate rows based on key columns
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged and cleaned dataset
        
    Returns
    -------
    bool
        True if all validations pass
        
    Raises
    ------
    ValueError
        If any validation check fails
        
    Examples
    --------
    >>> validate_data(df)
    True
    """
    print("Validating data integrity...")
    
    checks = {
        "date_not_null": df["date"].notna().all(),
        "district_not_null": df["district"].notna().all(),
        "pincode_not_null": df["pincode"].notna().all(),
        "no_duplicates": not df.duplicated(subset=KEY_COLUMNS).any()
    }
    
    failed = [k for k, v in checks.items() if not v]
    
    if failed:
        raise ValueError(
            f"Data validation failed: {', '.join(failed)}\n"
            "Please check your input data for issues."
        )
    
    print("✓ All validation checks passed")
    return True


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the merged dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset
        
    Returns
    -------
    dict
        Summary statistics including row count, date range, states, etc.
    """
    summary = {
        'total_rows': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'num_states': df['state'].nunique(),
        'num_districts': df['district'].nunique(),
        'num_pincodes': df['pincode'].nunique(),
        'states': sorted(df['state'].unique())
    }
    
    return summary
