"""
Data loading utilities for Aadhaar analysis.

This module handles loading CSV files and basic data formatting.
"""
import pandas as pd
from typing import Tuple
from config import DEMO_PATH, BIO_PATH, ENROLL_PATH
from constants import DATE_FORMAT


def load_aadhaar_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three Aadhaar datasets with error handling.
    
    Returns
    -------
    tuple of pd.DataFrame
        (demo, bio, enroll) dataframes
        
    Raises
    ------
    FileNotFoundError
        If any dataset file is missing
    ValueError
        If any dataset file is empty
    RuntimeError
        For other loading errors
        
    Examples
    --------
    >>> demo, bio, enroll = load_aadhaar_data()
    >>> print(f"Loaded {len(demo)} demo records")
    """
    datasets = {}
    paths = {
        'demo': DEMO_PATH,
        'bio': BIO_PATH,
        'enroll': ENROLL_PATH
    }
    
    for name, path in paths.items():
        try:
            datasets[name] = pd.read_csv(path)
            print(f"✓ Loaded {name} data: {len(datasets[name]):,} rows")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{name.capitalize()} data not found at {path}. "
                "Please check AADHAAR_DATA_PATH environment variable."
            )
        except pd.errors.EmptyDataError:
            raise ValueError(f"{name.capitalize()} data file is empty: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {name} data: {e}")
    
    return datasets['demo'], datasets['bio'], datasets['enroll']


def format_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Convert date column to datetime format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str, optional
        Name of date column (default: 'date')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with formatted date column
        
    Raises
    ------
    ValueError
        If date column is missing or cannot be parsed
        
    Examples
    --------
    >>> df = format_dates(df)
    >>> print(df['date'].dtype)
    datetime64[ns]
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    try:
        df[date_col] = pd.to_datetime(df[date_col], format=DATE_FORMAT)
        print(f"✓ Formatted {date_col} column to datetime")
    except Exception as e:
        raise ValueError(f"Error parsing dates: {e}")
    
    return df


def load_and_format_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets and format dates in one step.
    
    Returns
    -------
    tuple of pd.DataFrame
        (demo, bio, enroll) dataframes with formatted dates
        
    Examples
    --------
    >>> demo, bio, enroll = load_and_format_data()
    """
    demo, bio, enroll = load_aadhaar_data()
    
    demo = format_dates(demo)
    bio = format_dates(bio)
    enroll = format_dates(enroll)
    
    return demo, bio, enroll
