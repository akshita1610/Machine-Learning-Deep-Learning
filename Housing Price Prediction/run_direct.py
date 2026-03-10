import sys
sys.path.append('utils')
from data_loader import HousingDataLoader, load_sample_data
from model_utils import ModelTrainer
import pandas as pd
import numpy as np

print("=== Housing Price Prediction - Direct Run ===")

# Load and prepare data
loader = HousingDataLoader()
df = load_sample_data()
df = loader.clean_data(df)
df = loader.encode_categorical(df)
df = loader.create_features(df)
X, y = loader.prepare_features(df)

# Train models
trainer = ModelTrainer()
trainer.initialize_models()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and get results
results = trainer.train_models(X_train, y_train, X_test, y_test)

print("\nResults:")
for name, metrics in results.items():
    print(f"{name}: R² = {metrics['R²']:.4f}")

print("\n=== Direct Run Completed Successfully! ===")
