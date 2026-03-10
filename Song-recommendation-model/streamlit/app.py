# SmartTune Streamlit App - Fixed Version (No Matplotlib)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SmartTune - AI Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1DB954;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1DB954;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# DATA LOADING
# ================================

@st.cache_data
def load_data():
    """Load the song dataset."""
    data_path = Path('data/processed/feature_matrix.csv')
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    else:
        # Create sample data if not found
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
        
        return df

@st.cache_resource
def load_models():
    """Load trained models."""
    models = {}
    
    # Try to load baseline model
    baseline_path = Path('outputs/models/baseline_recommender.pkl')
    if baseline_path.exists():
        try:
            with open(baseline_path, 'rb') as f:
                models['baseline'] = pickle.load(f)
        except:
            pass
    
    # Try to load neural network model
    neural_path = Path('outputs/models/neural_recommender.pkl')
    if neural_path.exists():
        try:
            with open(neural_path, 'rb') as f:
                models['neural'] = pickle.load(f)
        except:
            pass
    
    return models

# ================================
# RECOMMENDATION ENGINE
# ================================

class SmartTuneEngine:
    """Simplified recommendation engine for Streamlit."""
    
    def __init__(self, data, models):
        self.data = data
        self.models = models
        
        # Create metadata with fallback
        if 'name' in data.columns and 'artist' in data.columns:
            self.metadata = data[['name', 'artist']].copy()
        else:
            self.metadata = pd.DataFrame({
                'name': [f'Track {i}' for i in range(len(data))],
                'artist': [f'Artist {i%10}' for i in range(len(data))]
            })
        
        # Create simple similarity matrix
        self._create_simple_similarity()
    
    def _create_simple_similarity(self):
        """Create simple cosine similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        
        # Get numeric features
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        feature_data = self.data[numeric_features]
        
        # Scale and compute similarity
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        self.similarity_matrix = cosine_similarity(scaled_features)
    
    def get_recommendations(self, track_name, model_type='hybrid', top_n=10):
        """Get recommendations for a track."""
        # Find the track
        matches = self.metadata[self.metadata['name'].str.contains(track_name, case=False, na=False)]
        
        if len(matches) == 0:
            return pd.DataFrame()
        
        track_idx = matches.index[0]
        
        # Use simple similarity
        similarities = self.similarity_matrix[track_idx]
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        recommendations = pd.DataFrame({
            'name': self.metadata.iloc[similar_indices]['name'].values,
            'artist': self.metadata.iloc[similar_indices]['artist'].values,
            'similarity_score': similarities[similar_indices]
        })
        return recommendations

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_feature_visualization(data):
    """Create feature distribution visualizations using Plotly."""
    key_features = ['danceability', 'energy', 'valence', 'tempo']
    available_features = [f for f in key_features if f in data.columns]
    
    if len(available_features) == 0:
        return None
    
    # Create subplots
    fig = go.Figure()
    
    for i, feature in enumerate(available_features[:4]):
        fig.add_trace(go.Histogram(
            x=data[feature],
            name=feature.replace('_', ' ').title(),
            nbinsx=20,
            opacity=0.7,
            marker_color='#1DB954'
        ))
    
    fig.update_layout(
        title='Audio Features Distribution',
        xaxis_title='Value',
        yaxis_title='Frequency',
        barmode='overlay',
        height=500
    )
    
    return fig

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 SmartTune</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Music Recommendation System</p>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading SmartTune engine..."):
        data = load_data()
        models = load_models()
        engine = SmartTuneEngine(data, models)
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">🎛️ Controls</h2>', unsafe_allow_html=True)
    
    # Model selection
    available_models = ['hybrid']
    if 'baseline' in models:
        available_models.append('baseline')
    if 'neural' in models:
        available_models.append('neural')
    
    model_type = st.sidebar.selectbox(
        "Recommendation Model",
        available_models,
        index=0
    )
    
    # Number of recommendations
    top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 10)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Get Song Recommendations</h2>', unsafe_allow_html=True)
        
        # Track selection
        track_names = engine.metadata['name'].tolist()
        selected_track = st.selectbox(
            "Select a song to get recommendations:",
            track_names,
            index=0
        )
        
        # Search option
        use_search = st.checkbox("Or search for a song")
        if use_search:
            search_query = st.text_input("Enter song name:")
            if search_query:
                matching_tracks = [track for track in track_names if search_query.lower() in track.lower()]
                if matching_tracks:
                    selected_track = st.selectbox("Matching tracks:", matching_tracks)
                else:
                    st.warning("No matching tracks found!")
        
        # Get recommendations button
        if st.button("🚀 Get Recommendations", type="primary"):
            with st.spinner("Finding similar songs..."):
                recommendations = engine.get_recommendations(selected_track, model_type, top_n)
            
            # Debug information
            st.write(f"Debug: Selected track = {selected_track}")
            st.write(f"Debug: Recommendations shape = {recommendations.shape}")
            st.write(f"Debug: Recommendations columns = {list(recommendations.columns)}")
            
            if len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} similar songs!")
                
                # Display query track
                query_idx = engine.metadata[engine.metadata['name'] == selected_track].index[0]
                query_artist = engine.metadata.iloc[query_idx]['artist']
                st.markdown(f"**🎵 Query:** {selected_track} by {query_artist}")
                
                # Display recommendations
                st.markdown("### 🎯 Recommendations")
                
                # Create a better display format
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1DB954; margin: 0.5rem 0;">
                        <h4 style="margin: 0; color: #1DB954;">{i}. {row['name']}</h4>
                        <p style="margin: 0.5rem 0; color: #666;"><strong>Artist:</strong> {row['artist']}</p>
                        <p style="margin: 0; color: #666;"><strong>Similarity Score:</strong> {row['similarity_score']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Also show as a dataframe for better visibility
                st.markdown("### 📋 Recommendation Table")
                st.dataframe(recommendations[['name', 'artist', 'similarity_score']], use_container_width=True)
            else:
                st.error("No recommendations found!")
                st.write("Debug: This might be because the similarity matrix couldn't be computed properly.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Music Analytics</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Songs", len(data))
        with col2:
            st.metric("Total Artists", engine.metadata['artist'].nunique())
        with col3:
            st.metric("Audio Features", len([col for col in data.columns if col not in ['track_id', 'name', 'artist']]))
        with col4:
            st.metric("Available Models", len(models))
        
        # Feature distributions
        st.markdown("### 📊 Feature Distributions")
        fig = create_feature_visualization(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Popular tracks
        st.markdown("### 🏆 Popular Tracks")
        if 'popularity' in data.columns:
            popular_tracks = data.nlargest(10, 'popularity')[['name', 'artist', 'popularity']]
            st.dataframe(popular_tracks, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">About SmartTune</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎵 What is SmartTune?
        
        SmartTune is an AI-powered music recommendation system that suggests songs based on their musical characteristics.
        
        ### 🧠 How It Works
        
        1. **Feature Analysis**: Extracts and analyzes audio features from songs
        2. **Machine Learning**: Uses similarity algorithms to find matching songs
        3. **Real-Time Recommendations**: Delivers instant song suggestions
        4. **Interactive Interface**: Easy-to-use web application
        
        ### 📊 Audio Features Analyzed
        
        - **Danceability**: How suitable for dancing
        - **Energy**: Perceived intensity and activity
        - **Valence**: Musical positiveness (happy vs sad)
        - **Tempo**: Speed in beats per minute
        - **Acousticness**: Confidence that track is acoustic
        - **Instrumentalness**: Confidence track has no vocals
        - **Liveness**: Presence of audience in recording
        - **Speechiness**: Presence of spoken words
        
        ---
        
        *Built with ❤️ for music lovers everywhere*
        """)

if __name__ == "__main__":
    main()
