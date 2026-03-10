"""
Ames Housing Dataset Download Helper

This script provides information about downloading the Ames Housing Dataset
for the Housing Price Prediction project.
"""

import os
import urllib.request
import pandas as pd

def check_dataset_exists():
    """Check if the Ames Housing Dataset already exists."""
    data_path = os.path.join('data', 'AmesHousing.csv')
    return os.path.exists(data_path)

def get_dataset_info():
    """Get information about the Ames Housing Dataset."""
    info = {
        'name': 'Ames Housing Dataset',
        'source': 'Kaggle',
        'url': 'https://www.kaggle.com/datasets/prevek18/ames-housing-dataset',
        'size': '~2.5 MB',
        'records': '2,930 houses',
        'features': '82 features including SalePrice',
        'file_name': 'AmesHousing.csv'
    }
    return info

def print_download_instructions():
    """Print step-by-step download instructions."""
    info = get_dataset_info()
    
    print("=" * 60)
    print("Ames Housing Dataset Download Instructions")
    print("=" * 60)
    print()
    print("Dataset Information:")
    print(f"  Name: {info['name']}")
    print(f"  Size: {info['size']}")
    print(f"  Records: {info['records']}")
    print(f"  Features: {info['features']}")
    print()
    print("Download Steps:")
    print("1. Visit: " + info['url'])
    print("2. Sign in to Kaggle (free account required)")
    print("3. Click the 'Download' button")
    print("4. Save 'AmesHousing.csv' to the 'data/' folder")
    print()
    print("File Location:")
    print("  Place the file here: data/AmesHousing.csv")
    print()
    print("After downloading:")
    print("  - Restart the Jupyter notebook")
    print("  - The notebook will automatically detect and use the real dataset")
    print("  - You'll see much better model performance (R² ~0.8-0.9)")
    print()
    print("=" * 60)

def verify_dataset():
    """Verify the dataset is properly placed and formatted."""
    data_path = os.path.join('data', 'AmesHousing.csv')
    
    if not os.path.exists(data_path):
        print("❌ Dataset not found at:", data_path)
        return False
    
    try:
        # Try to read the dataset
        df = pd.read_csv(data_path)
        
        # Check for expected columns
        expected_columns = ['SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print("⚠️  Warning: Missing expected columns:", missing_columns)
        
        print("✅ Dataset verified successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   SalePrice range: ${df['SalePrice'].min():,.0f} - ${df['SalePrice'].max():,.0f}")
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def main():
    """Main function to guide dataset download."""
    print("Housing Price Prediction - Dataset Setup")
    print()
    
    if check_dataset_exists():
        print("Dataset already found!")
        verify_dataset()
    else:
        print("Dataset not found. Download required.")
        print()
        print_download_instructions()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        print("Data directory created/verified")

if __name__ == "__main__":
    main()
