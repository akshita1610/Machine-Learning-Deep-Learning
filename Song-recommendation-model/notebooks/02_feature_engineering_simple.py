# SmartTune Step 3: Feature Engineering

"""
Step 3: Feature Engineering for SmartTune Recommendation System

This notebook transforms raw audio features into machine learning-ready features
through scaling, engineering, and selection techniques.
"""

# Core data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Load the processed dataset from Step 2
data_path = Path('data/processed/sample_tracks_processed.csv')

if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
else:
    print("Dataset not found. Creating sample data for demonstration...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'track_id': [f'track_{i}' for i in range(n_samples)],
        'name': [f'Track {i}' for i in range(n_samples)],
        'artist': [f'Artist {i%10}' for i in range(n_samples)],
        'danceability': np.random.beta(2, 2, n_samples),
        'energy': np.random.beta(2, 1.5, n_samples),
        'valence': np.random.beta(2, 2, n_samples),
        'tempo': np.random.normal(120, 20, n_samples),
        'acousticness': np.random.beta(1, 3, n_samples),
        'instrumentalness': np.random.beta(1, 10, n_samples),
        'liveness': np.random.beta(1, 5, n_samples),
        'speechiness': np.random.beta(1, 8, n_samples),
        'key': np.random.randint(0, 12, n_samples),
        'mode': np.random.randint(0, 2, n_samples),
        'loudness': np.random.normal(-10, 5, n_samples),
        'duration_ms': np.random.normal(210000, 30000, n_samples),
        'popularity': np.random.randint(0, 100, n_samples)
    })
    
    # Ensure values are in valid ranges
    df['tempo'] = np.clip(df['tempo'], 50, 200)
    df['loudness'] = np.clip(df['loudness'], -60, 0)
    df['duration_ms'] = np.clip(df['duration_ms'], 30000, 600000)
    
    print(f"Created sample dataset: {df.shape}")

print("\nDataset Preview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# STEP 1: FEATURE SCALING
print("\n" + "="*60)
print("STEP 1: FEATURE SCALING")
print("="*60)

class FeatureScaler:
    """Custom feature scaler for audio features."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_info = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, features: list):
        """Fit scalers to the data."""
        X = df[features].copy()
        
        for feature in features:
            feature_data = X[feature]
            
            # Determine scaling strategy based on feature characteristics
            if feature in ['danceability', 'energy', 'valence', 'acousticness', 
                          'instrumentalness', 'liveness', 'speechiness']:
                scaler = MinMaxScaler()
                scaling_type = 'minmax'
            elif feature in ['tempo', 'duration_ms']:
                scaler = RobustScaler()
                scaling_type = 'robust'
            elif feature == 'loudness':
                scaler = RobustScaler()
                scaling_type = 'robust'
            elif feature in ['key', 'mode']:
                scaler = None
                scaling_type = 'categorical'
            else:
                scaler = StandardScaler()
                scaling_type = 'standard'
            
            self.scalers[feature] = scaler
            self.feature_info[feature] = {
                'type': scaling_type,
                'min': feature_data.min(),
                'max': feature_data.max(),
                'mean': feature_data.mean(),
                'std': feature_data.std()
            }
            
            if scaler is not None:
                scaler.fit(feature_data.values.reshape(-1, 1))
        
        self.is_fitted = True
        print(f"FeatureScaler fitted on {len(features)} features")
    
    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Transform features using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transformation")
        
        X = df[features].copy()
        X_scaled = X.copy()
        
        for feature in features:
            scaler = self.scalers[feature]
            if scaler is not None:
                X_scaled[feature] = scaler.transform(X[feature].values.reshape(-1, 1)).flatten()
        
        return X_scaled
    
    def fit_transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)

# Apply scaling
audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                 'instrumentalness', 'liveness', 'speechiness', 'key', 'mode', 
                 'loudness', 'duration_ms']

available_features = [f for f in audio_features if f in df.columns]
scaler = FeatureScaler()
df_scaled = scaler.fit_transform(df, available_features)

print("\nFeature scaling complete!")
print(f"Scaled {len(available_features)} features")

# STEP 2: FEATURE ENGINEERING
print("\n" + "="*60)
print("STEP 2: FEATURE ENGINEERING")
print("="*60)

class FeatureEngineer:
    """Create engineered features from audio features."""
    
    def __init__(self):
        self.engineered_features = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        df_eng = df.copy()
        
        # Energy features
        df_eng['energy_danceability'] = df_eng['energy'] * df_eng['danceability']
        df_eng['energy_valence'] = df_eng['energy'] * df_eng['valence']
        df_eng['high_energy'] = (df_eng['energy'] > 0.7).astype(int)
        
        # Mood features
        df_eng['mood_score'] = (df_eng['valence'] + df_eng['energy']) / 2
        df_eng['is_happy'] = (df_eng['valence'] > 0.7).astype(int)
        df_eng['is_sad'] = ((df_eng['valence'] < 0.3) & (df_eng['energy'] < 0.5)).astype(int)
        
        # Rhythm features
        df_eng['tempo_slow'] = (df_eng['tempo'] < 90).astype(int)
        df_eng['tempo_fast'] = (df_eng['tempo'] >= 140).astype(int)
        df_eng['rhythm_complexity'] = df_eng['danceability'] * (df_eng['tempo'] / 200)
        
        # Genre features
        df_eng['is_electronic'] = ((df_eng['acousticness'] < 0.2) & (df_eng['energy'] > 0.6)).astype(int)
        df_eng['is_acoustic'] = ((df_eng['acousticness'] > 0.7) & (df_eng['energy'] < 0.5)).astype(int)
        df_eng['is_pop'] = ((df_eng['danceability'] > 0.6) & (df_eng['valence'] > 0.5)).astype(int)
        
        self.engineered_features = [
            'energy_danceability', 'energy_valence', 'high_energy',
            'mood_score', 'is_happy', 'is_sad',
            'tempo_slow', 'tempo_fast', 'rhythm_complexity',
            'is_electronic', 'is_acoustic', 'is_pop'
        ]
        
        print(f"Created {len(self.engineered_features)} engineered features")
        return df_eng

# Apply feature engineering
feature_engineer = FeatureEngineer()
df_engineered = feature_engineer.create_all_features(df_scaled)

print("\nFeature Engineering Complete!")
print(f"Original: {len(available_features)}, Engineered: {len(feature_engineer.engineered_features)}")

print("\nSample Engineered Features:")
print(df_engineered[feature_engineer.engineered_features[:5]].head())

# STEP 3: FEATURE SELECTION
print("\n" + "="*60)
print("STEP 3: FEATURE SELECTION")
print("="*60)

class FeatureSelector:
    """Select the most important features for recommendations."""
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_col: str = 'popularity'):
        """Analyze feature importance using statistical methods."""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Using variance analysis instead.")
            target_col = None
        
        feature_scores = {}
        
        for feature in numeric_features:
            if feature == target_col:
                continue
                
            if target_col and target_col in df.columns:
                correlation = abs(df[feature].corr(df[target_col]))
                feature_scores[feature] = correlation
            else:
                variance = df[feature].var()
                feature_scores[feature] = variance
        
        self.feature_scores = feature_scores
        return feature_scores
    
    def select_top_features(self, df: pd.DataFrame, n_features: int = 20) -> list:
        """Select top N features based on importance scores."""
        if not self.feature_scores:
            self.analyze_feature_importance(df)
        
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [feature for feature, score in sorted_features[:n_features]]
        
        return self.selected_features

# Apply feature selection
feature_selector = FeatureSelector()
feature_scores = feature_selector.analyze_feature_importance(df_engineered)
top_features = feature_selector.select_top_features(df_engineered, n_features=20)

print("\nFeature Selection Complete!")
print(f"Analyzed {len(feature_scores)} features")
print(f"Selected top {len(top_features)} features")

print("\nTop 10 Features:")
for i, feature in enumerate(top_features[:10], 1):
    score = feature_scores.get(feature, 0)
    print(f"{i:2d}. {feature:25s} (score: {score:.3f})")

# STEP 4: SAVE RESULTS
print("\n" + "="*60)
print("STEP 4: SAVE RESULTS")
print("="*60)

# Create output directory
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

# Create final feature matrix
all_features = available_features + feature_engineer.engineered_features
# Make sure we only use features that exist in the dataframe
existing_features = [f for f in all_features if f in df_engineered.columns]
id_features = ['track_id', 'name', 'artist']
existing_id_features = [f for f in id_features if f in df_engineered.columns]

feature_matrix = df_engineered[existing_features + existing_id_features].copy()

# Save feature matrix
feature_matrix_path = output_dir / 'feature_matrix.csv'
feature_matrix.to_csv(feature_matrix_path, index=False)
print(f"Feature matrix saved: {feature_matrix_path}")

# Save feature engineering metadata
import json

metadata = {
    'original_features': available_features,
    'engineered_features': feature_engineer.engineered_features,
    'selected_features': top_features,
    'feature_scores': feature_scores,
    'total_features': len(all_features)
}

metadata_path = output_dir / 'feature_engineering_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {metadata_path}")

# SUMMARY
print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)

print("What We Accomplished:")
print("1. Scaled Features: Applied appropriate scaling to normalize ranges")
print("2. Engineered Features: Created 12 new features capturing musical patterns")
print("3. Feature Selection: Identified top 20 most important features")
print("4. Saved Results: Stored engineered features and metadata")

print(f"\nFinal Results:")
print(f"• Original features: {len(available_features)}")
print(f"• Engineered features: {len(feature_engineer.engineered_features)}")
print(f"• Selected features: {len(top_features)}")
print(f"• Final matrix shape: {feature_matrix.shape}")

print("\nReady for Step 4: Model Training!")
print("Next we'll build baseline and neural network recommendation models")

print("\nFeature Engineering Complete!")
