"""
Visualization utilities for Aadhaar analysis.

This module provides reusable plotting functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Optional, List
from config import MAP_PATH
from constants import DEFAULT_COLORMAP, DEFAULT_FIGURE_SIZE, MAP_FIGURE_SIZE

# Cache for map data to avoid repeated file reads
_map_cache = None


def get_india_map() -> gpd.GeoDataFrame:
    """
    Load India map GeoJSON (cached for performance).
    
    Returns
    -------
    gpd.GeoDataFrame
        India map with state boundaries
    """
    global _map_cache
    
    if _map_cache is None:
        _map_cache = gpd.read_file(MAP_PATH)
        _map_cache["state"] = _map_cache["STNAME"].str.lower().str.strip()
        print("✓ Loaded India map (cached)")
    
    return _map_cache.copy()


def plot_state_map(
    data: pd.DataFrame, 
    col: str, 
    title: str, 
    colorScheme: str = DEFAULT_COLORMAP
) -> None:
    """
    Plot choropleth map of Indian states.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'state' column and data to visualize
    col : str
        Column name to visualize on the map
    title : str
        Plot title
    colorScheme : str, optional
        Matplotlib colormap name (default from constants)
        
    Raises
    ------
    ValueError
        If required columns are missing from data
        
    Examples
    --------
    >>> state_data = df.groupby('state')['total_activity'].sum().reset_index()
    >>> plot_state_map(state_data, 'total_activity', 'Total Activity by State')
    """
    # Validate inputs
    if 'state' not in data.columns:
        raise ValueError("Data must contain 'state' column")
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in data")
    
    # Load and merge map data
    india_map = get_india_map()
    india_merged = india_map.merge(data, on="state", how="left")
    
    # Create plot
    fig, ax = plt.subplots(figsize=MAP_FIGURE_SIZE)
    india_merged.plot(
        column=col,
        cmap=colorScheme, 
        linewidth=0.6, 
        edgecolor="black",
        legend=True, 
        ax=ax,
        missing_kwds={'color': 'lightgrey'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis("off")
    ax.ticklabel_format(style="plain", axis="y")
    
    plt.tight_layout()
    plt.show()


def plot_bar(
    data: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    horizontal: bool = False,
    top_n: Optional[int] = None,
    color: Optional[str] = None,
    figsize: tuple = DEFAULT_FIGURE_SIZE
) -> None:
    """
    Create bar plot from data.
    
    Can handle both Series (when x_col and y_col are None) and
    DataFrame with specified columns.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Data to plot
    x_col : str, optional
        X-axis column name (for DataFrame)
    y_col : str, optional
        Y-axis column name (for DataFrame)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    horizontal : bool
        If True, create horizontal bar plot
    top_n : int, optional
        Show only top N values (sorted by y values)
    color : str, optional
        Bar color
    figsize : tuple, optional
        Figure size (width, height)
        
    Examples
    --------
    >>> # From Series
    >>> state_counts = df['state'].value_counts()
    >>> plot_bar(state_counts, title='Records by State', top_n=10)
    
    >>> # From DataFrame
    >>> plot_bar(df, x_col='state', y_col='total_activity', 
    ...          title='Activity by State', horizontal=True)
    """
    # Handle Series or DataFrame without column specification
    if x_col is None and y_col is None:
        if isinstance(data, pd.Series):
            data = data.sort_values(ascending=False)
            if top_n is not None:
                data = data.head(top_n)
            x = data.index
            y = data.values
        else:
            raise ValueError(
                "For DataFrame, specify x_col and y_col, "
                "or pass a Series instead"
            )
    else:
        # Handle DataFrame with column specification
        if x_col not in data.columns:
            raise ValueError(f"Column '{x_col}' not found in data")
        if y_col not in data.columns:
            raise ValueError(f"Column '{y_col}' not found in data")
        
        data = data.copy()
        if top_n is not None:
            data = data.sort_values(y_col, ascending=False).head(top_n)
        x = data[x_col]
        y = data[y_col]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    if horizontal:
        plt.barh(x, y, color=color)
        plt.gca().invert_yaxis()  # Highest value at top
    else:
        plt.bar(x, y, color=color)
        if len(x) > 10:  # Rotate labels if many categories
            plt.xticks(rotation=45, ha='right')
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y' if not horizontal else 'x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def clear_map_cache():
    """
    Clear the cached map data.
    
    Useful if you need to reload the map file or free memory.
    """
    global _map_cache
    _map_cache = None
    print("✓ Map cache cleared")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (2D array)
    class_names : list of str
        Names of the classes
    title : str, optional
        Plot title (default: "Confusion Matrix")
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap name (default: 'Blues')
        
    Examples
    --------
    >>> from utils.ml_models import train_activity_classifier
    >>> results = train_activity_classifier(df)
    >>> cm = results['metrics']['confusion_matrix']
    >>> class_names = ['Enrolment', 'Biometric', 'Demographic']
    >>> plot_confusion_matrix(cm, class_names)
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True,           # Show numbers
        fmt='d',              # Integer format
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
