"""
Constants for Aadhaar data analysis.

This module defines all constant values, codes, and configuration
parameters used throughout the analysis.
"""

# Unit conversions
LAKH = 100_000
CRORE = 10_000_000

# Activity type codes
ACTIVITY_ENROLMENT = 0
ACTIVITY_BIOMETRIC = 1
ACTIVITY_DEMOGRAPHIC = 2
ACTIVITY_NONE = -1

ACTIVITY_NAMES = {
    ACTIVITY_ENROLMENT: "Enrolment",
    ACTIVITY_BIOMETRIC: "Biometric Update",
    ACTIVITY_DEMOGRAPHIC: "Demographic Update",
    ACTIVITY_NONE: "No Activity"
}

ACTIVITY_COLUMN_MAP = {
    "total_enrolments": ACTIVITY_ENROLMENT,
    "total_bio_updates": ACTIVITY_BIOMETRIC,
    "total_demo_updates": ACTIVITY_DEMOGRAPHIC
}

# Day of week constants
MONDAY = 0
TUESDAY = 1
WEDNESDAY = 2
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6

WEEKEND_DAYS = [SATURDAY, SUNDAY]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Key columns for merging datasets
KEY_COLUMNS = ["date", "state", "district", "pincode"]

# Numeric columns that need to be filled with 0 for NaN values
NUMERIC_COLUMNS = [
    "demo_age_5_17", 
    "demo_age_17_",
    "bio_age_5_17", 
    "bio_age_17_",
    "age_0_5", 
    "age_5_17", 
    "age_18_greater"
]

# Age group labels
AGE_GROUP_0_5 = "0-5"
AGE_GROUP_5_17 = "5-17"
AGE_GROUP_18_PLUS = "18+"

# Date format used in CSV files
DATE_FORMAT = '%d-%m-%Y'

# Default visualization settings
DEFAULT_COLORMAP = 'jet'
DEFAULT_FIGURE_SIZE = (10, 6)
MAP_FIGURE_SIZE = (8, 10)
