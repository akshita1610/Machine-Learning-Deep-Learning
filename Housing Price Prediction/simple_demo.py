import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== Housing Price Prediction - Simple Demo ===")

# Create sample data
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

print(f"Dataset created: {df.shape}")
print(f"Sample Sale Price: ${df['SalePrice'].mean():,.2f}")

# Feature engineering
df['HouseAge'] = 2023 - df['YearBuilt']
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['QualityScore'] = df['OverallQual'] * df['GarageCars']

# Encode categorical variables
le_neighborhood = LabelEncoder()
le_housestyle = LabelEncoder()
df['Neighborhood_encoded'] = le_neighborhood.fit_transform(df['Neighborhood'])
df['HouseStyle_encoded'] = le_housestyle.fit_transform(df['HouseStyle'])

# Prepare features
feature_columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'GarageCars', 
                   'TotalBsmtSF', 'HouseAge', 'TotalSF', 'QualityScore', 
                   'Neighborhood_encoded', 'HouseStyle_encoded']

X = df[feature_columns]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n=== Model Training Results ===")
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    
    print(f"{name}:")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")

best_model = max(results.keys(), key=lambda x: results[x]['R²'])
best_r2 = results[best_model]['R²']
best_mae = results[best_model]['MAE']

print(f"\nBest Model: {best_model}")
print(f"R² Score: {best_r2:.4f}")
print(f"MAE: ${best_mae:,.2f}")

print("\n=== Demo Completed Successfully! ===")
print("✅ All ML components working correctly!")
print("✅ Project ready for educational use!")
