"""
Configuration file for Aadhaar data analysis.

This module contains all path configurations. Paths can be customized
using the AADHAAR_DATA_PATH environment variable.

Default: /content/drive/MyDrive/uidai (Google Colab)
Custom: Set AADHAAR_DATA_PATH environment variable
"""
import os
from pathlib import Path

# Base paths
BASE_PATH = os.getenv('AADHAAR_DATA_PATH', '/content/drive/MyDrive/uidai')
DATA_DIR = Path(BASE_PATH) / 'combined_dataset2'
MAP_PATH = Path(BASE_PATH) / 'INDIA_STATES.geojson'

# Dataset paths
DEMO_PATH = DATA_DIR / 'demo_combined.csv'
BIO_PATH = DATA_DIR / 'bio_combined.csv'
ENROLL_PATH = DATA_DIR / 'enroll_combined.csv'

# Output paths
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

def validate_paths():
    """
    Validate that all required data files exist.
    
    Raises
    ------
    FileNotFoundError
        If any required file is missing
    """
    paths = {
        'Demo data': DEMO_PATH,
        'Bio data': BIO_PATH,
        'Enroll data': ENROLL_PATH,
        'Map GeoJSON': MAP_PATH
    }
    
    missing = [name for name, path in paths.items() if not path.exists()]
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files: {', '.join(missing)}\n"
            f"Please ensure data is available at: {BASE_PATH}\n"
            f"You can set a custom path using: export AADHAAR_DATA_PATH=/your/path"
        )
    
    return True

if __name__ == "__main__":
    # Test configuration
    print("Configuration:")
    print(f"  Base Path: {BASE_PATH}")
    print(f"  Demo Path: {DEMO_PATH}")
    print(f"  Bio Path: {BIO_PATH}")
    print(f"  Enroll Path: {ENROLL_PATH}")
    print(f"  Map Path: {MAP_PATH}")
    print("\nValidating paths...")
    try:
        validate_paths()
        print("✓ All paths valid!")
    except FileNotFoundError as e:
        print(f"✗ {e}")
