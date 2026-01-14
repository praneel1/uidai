# Aadhaar Data Analysis

Analysis of Aadhaar enrolment and update patterns across India to identify meaningful trends and insights.

## Problem Statement

Unlocking Societal Trends in Aadhaar Enrolment and Updates - Identify meaningful patterns, trends, anomalies, or predictive indicators in Aadhaar data and translate them into clear insights or solution frameworks that can support informed decision-making and system improvements.

## Datasets

This analysis uses three types of Aadhaar data:

### 1. Demographic Updates (demo_combined.csv)
Count of demographic information updates by age group.

### 2. Biometric Updates (bio_combined.csv)
Count of biometric (fingerprint/iris) updates by age group.

### 3. Enrolments (enroll_combined.csv)
Count of new Aadhaar enrolments by age group.

### 4. Geographic Boundaries (INDIA_STATES.geojson)
GeoJSON file containing state boundary polygons for map visualizations.

### Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Date of activity (DD-MM-YYYY format) |
| state | string | State name (lowercase) |
| district | string | District name |
| pincode | integer | 6-digit postal code |
| demo_age_5_17 | integer | Demographic updates for ages 5-17 |
| demo_age_17_ | integer | Demographic updates for ages 17+ |
| bio_age_5_17 | integer | Biometric updates for ages 5-17 |
| bio_age_17_ | integer | Biometric updates for ages 17+ |
| age_0_5 | integer | New enrolments for ages 0-5 |
| age_5_17 | integer | New enrolments for ages 5-17 |
| age_18_greater | integer | New enrolments for ages 18+ |

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up data path (optional - defaults to `/content/drive/MyDrive/uidai` for Google Colab):
   
   ```bash
   # Linux/Mac
   export AADHAAR_DATA_PATH="/path/to/your/data"
   
   # Windows (PowerShell)
   $env:AADHAAR_DATA_PATH="C:\path\to\your\data"
   
   # Windows (CMD)
   set AADHAAR_DATA_PATH=C:\path\to\your\data
   ```

4. Organize your data:
   ```
   your_data_path/
   ├── combined_dataset2/
   │   ├── demo_combined.csv
   │   ├── bio_combined.csv
   │   └── enroll_combined.csv
   └── INDIA_STATES.geojson
   ```

5. Run the analysis:
   ```bash
   jupyter notebook
   ```

## Project Structure

```
aadhaar-analysis/
├── config.py                          # Configuration and paths
├── constants.py                       # Constants and definitions
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── CODE_AUDIT_REPORT.md              # Code quality audit
├── utils/                            # Utility modules
│   ├── __init__.py
│   ├── data_loader.py                # Data loading functions
│   ├── preprocessing.py              # Data cleaning & merging
│   ├── features.py                   # Feature engineering
│   ├── visualizations.py             # Plotting functions
│   └── ml_models.py                  # Machine learning models
├── Clean_analiz.ipynb                # Analysis notebook
├── Clean_analiz_draft2.ipynb         # Draft version
└── Clean_analiz_3_pre_final.ipynb    # Pre-final version
```

## Usage

### Using Utility Functions

```python
# Import utilities
from utils import (
    load_aadhaar_data,
    format_dates,
    merge_datasets,
    clean_merged_data,
    add_all_features,
    plot_state_map,
    plot_bar
)

# Load and prepare data
demo, bio, enroll = load_aadhaar_data()
demo = format_dates(demo)
bio = format_dates(bio)
enroll = format_dates(enroll)

# Merge and clean
df = merge_datasets(demo, bio, enroll)
df = clean_merged_data(df)

# Add features
df = add_all_features(df)

# Visualize
state_totals = df.groupby('state')['total_activity'].sum().reset_index()
plot_state_map(state_totals, 'total_activity', 'Total Activity by State')
```

### Machine Learning

```python
from utils import train_all_models

# Train all ML models
results = train_all_models(df)

# Access results
print(f"Regression R²: {results['regression']['metrics']['r2_test']:.4f}")
print(f"Classification Accuracy: {results['classification']['metrics']['accuracy_test']:.4f}")
print(f"Clustering Silhouette: {results['clustering']['metrics']['silhouette']:.4f}")
```

## Key Findings

[Add your key findings here after analysis]

1. **Finding 1:** [Description]
2. **Finding 2:** [Description]
3. **Finding 3:** [Description]

## Visualizations

The analysis includes:

1. Choropleth Maps - State-wise activity distribution
2. Bar Charts - Top states by activity type
3. Time Series - Daily activity trends
4. Heatmaps - Temporal patterns
5. Scatter Plots - District-level analysis

## Machine Learning Models

The analysis includes three ML models:

### 1. Linear Regression
- Purpose: Predict total activity based on temporal features
- Features: day, month, is_weekend
- Target: total_activity

### 2. Logistic Regression
- Purpose: Classify dominant activity type
- Features: demo_ratio, bio_ratio, enrol_ratio, day_of_week, month
- Target: activity_type

### 3. K-Means Clustering
- Purpose: Group similar activity patterns
- Features: Standardized activity ratios and temporal features

## Testing

To test the utility functions:

```bash
# Test imports
python -c "from utils import *; print('All imports successful')"
```

---

**Last Updated:** January 14, 2026
