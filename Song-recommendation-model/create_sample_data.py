import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create sample music data for the Streamlit app."""
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
    
    # Add engineered features
    df['energy_danceability'] = df['energy'] * df['danceability']
    df['mood_score'] = (df['valence'] + df['energy']) / 2
    df['rhythm_complexity'] = df['danceability'] * (df['tempo'] / 200)
    df['is_electronic'] = ((df['acousticness'] < 0.2) & (df['energy'] > 0.6)).astype(int)
    df['is_acoustic'] = ((df['acousticness'] > 0.7) & (df['energy'] < 0.5)).astype(int)
    df['is_pop'] = ((df['danceability'] > 0.6) & (df['valence'] > 0.5)).astype(int)
    
    # Save the data
    output_path = Path('data/processed/feature_matrix.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created sample data with {len(df)} tracks")
    print(f"Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    create_sample_data()
