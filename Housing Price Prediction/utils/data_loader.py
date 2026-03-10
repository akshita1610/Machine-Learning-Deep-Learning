"""
Data Loading and Preprocessing Utilities

This module provides functions for loading, cleaning, and preprocessing
the housing dataset for machine learning tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class HousingDataLoader:
    """
    A class for loading and preprocessing housing data.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, file_path):
        """
        Load housing data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing housing data
            
        Returns:
        --------
        pd.DataFrame
            Loaded housing dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully! Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean the housing dataset by handling missing values and outliers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input housing dataset
            
        Returns:
        --------
        pd.DataFrame
            Cleaned housing dataset
        """
        print("Starting data cleaning...")
        original_shape = df.shape[0]
        
        # Handle missing values
        # Numerical columns - fill with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove outliers from key numerical columns
        key_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF']
        for feature in key_features:
            if feature in df.columns:
                df = self._remove_outliers(df, feature)
        
        print(f"Data cleaning completed!")
        print(f"Removed {original_shape - df.shape[0]} outliers")
        print(f"Final shape: {df.shape}")
        
        return df
    
    def _remove_outliers(self, df, column, method='iqr'):
        """
        Remove outliers using IQR method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        column : str
            Column name to remove outliers from
        method : str
            Method for outlier removal ('iqr')
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with outliers removed
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
    
    def encode_categorical(self, df):
        """
        Encode categorical variables using Label Encoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical variables
        """
        print("Encoding categorical variables...")
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        print(f"Encoded {len(categorical_columns)} categorical columns")
        return df
    
    def create_features(self, df):
        """
        Create new features from existing ones.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional engineered features
        """
        print("Creating engineered features...")
        
        # House age
        if 'YearBuilt' in df.columns:
            df['HouseAge'] = 2023 - df['YearBuilt']
        
        # Total square footage
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
        
        # Bathrooms per square foot
        if 'FullBath' in df.columns and 'GrLivArea' in df.columns:
            df['BathPerSF'] = df['FullBath'] / df['GrLivArea']
        
        # Bedrooms per square foot
        if 'BedroomAbvGr' in df.columns and 'GrLivArea' in df.columns:
            df['BedroomPerSF'] = df['BedroomAbvGr'] / df['GrLivArea']
        
        # Overall quality score
        if 'OverallQual' in df.columns and 'GarageCars' in df.columns:
            df['QualityScore'] = df['OverallQual'] * df['GarageCars']
        
        print("Feature engineering completed!")
        return df
    
    def prepare_features(self, df, target_column='SalePrice'):
        """
        Prepare features for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of the target variable column
            
        Returns:
        --------
        tuple
            (X, y) feature matrix and target vector
        """
        # Select numerical features (excluding target)
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_features:
            numerical_features.remove(target_column)
        
        # Add encoded categorical features
        encoded_categorical = [col for col in df.columns if col.endswith('_encoded')]
        
        # Combine features
        feature_columns = numerical_features + encoded_categorical
        self.feature_columns = feature_columns
        
        # Remove original categorical columns
        original_categorical = df.select_dtypes(include=['object']).columns
        feature_columns = [col for col in feature_columns if col not in original_categorical]
        
        X = df[feature_columns]
        y = df[target_column]
        
        print(f"Prepared {len(feature_columns)} features for modeling")
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale numerical features using StandardScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame, optional
            Test features
            
        Returns:
        --------
        tuple
            Scaled training and test features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled


def load_sample_data():
    """
    Create a sample dataset for demonstration purposes.
    
    Returns:
    --------
    pd.DataFrame
        Sample housing dataset
    """
    print("Creating sample dataset for demonstration...")
    np.random.seed(42)
    sample_size = 1000
    
    df = pd.DataFrame({
        'SalePrice': np.random.normal(200000, 50000, sample_size),
        'GrLivArea': np.random.normal(1500, 500, sample_size),
        'BedroomAbvGr': np.random.randint(1, 6, sample_size),
        'FullBath': np.random.randint(1, 4, sample_size),
        'YearBuilt': np.random.randint(1950, 2020, sample_size),
        'OverallQual': np.random.randint(1, 10, sample_size),
        'GarageCars': np.random.randint(0, 4, sample_size),
        'TotalBsmtSF': np.random.normal(1000, 400, sample_size),
        'Neighborhood': np.random.choice(['NorthAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'], sample_size),
        'HouseStyle': np.random.choice(['1Story', '2Story', '1.5Fin', 'SLvl'], sample_size)
    })
    
    # Ensure positive values
    df['GrLivArea'] = np.abs(df['GrLivArea'])
    df['TotalBsmtSF'] = np.abs(df['TotalBsmtSF'])
    df['SalePrice'] = np.abs(df['SalePrice'])
    
    print(f"Sample dataset created with shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Example usage
    loader = HousingDataLoader()
    
    # Try to load real data, fall back to sample if not found
    try:
        df = loader.load_data("../data/AmesHousing.csv")
        if df is None:
            df = load_sample_data()
    except:
        df = load_sample_data()
    
    # Process the data
    df = loader.clean_data(df)
    df = loader.encode_categorical(df)
    df = loader.create_features(df)
    
    X, y = loader.prepare_features(df)
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
