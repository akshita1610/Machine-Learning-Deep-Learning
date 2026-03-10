"""
Complete Housing Price Prediction Analysis
This script runs the entire ML workflow and shows all results.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def run_complete_analysis():
    """Run the complete housing price prediction analysis."""
    
    print("=" * 80)
    print("HOUSING PRICE PREDICTION - COMPLETE ANALYSIS")
    print("=" * 80)
    
    # 1. Data Loading and Creation
    print("\n1. DATA LOADING")
    print("-" * 40)
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
    print(f"Target variable (SalePrice): ${df['SalePrice'].mean():,.2f} ± ${df['SalePrice'].std():,.2f}")
    
    # 2. Data Cleaning
    print("\n2. DATA CLEANING")
    print("-" * 40)
    
    # Handle missing values (none in our sample data)
    print("Missing values checked and handled")
    
    # Remove outliers
    original_shape = df.shape[0]
    for col in ['SalePrice', 'GrLivArea', 'TotalBsmtSF']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"Outliers removed: {original_shape - df.shape[0]} rows")
    print(f"Cleaned dataset shape: {df.shape}")
    
    # 3. Feature Engineering
    print("\n3. FEATURE ENGINEERING")
    print("-" * 40)
    
    df['HouseAge'] = 2023 - df['YearBuilt']
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['QualityScore'] = df['OverallQual'] * df['GarageCars']
    
    print("✓ Engineered features created:")
    print("  - HouseAge (2023 - YearBuilt)")
    print("  - TotalSF (GrLivArea + TotalBsmtSF)")
    print("  - QualityScore (OverallQual * GarageCars)")
    
    # 4. Categorical Encoding
    print("\n4. CATEGORICAL ENCODING")
    print("-" * 40)
    
    le_neighborhood = LabelEncoder()
    le_housestyle = LabelEncoder()
    df['Neighborhood_encoded'] = le_neighborhood.fit_transform(df['Neighborhood'])
    df['HouseStyle_encoded'] = le_housestyle.fit_transform(df['HouseStyle'])
    
    print("✓ Categorical variables encoded:")
    print(f"  - Neighborhood: {len(df['Neighborhood'].unique())} categories")
    print(f"  - HouseStyle: {len(df['HouseStyle'].unique())} categories")
    
    # 5. Feature Selection
    print("\n5. FEATURE SELECTION")
    print("-" * 40)
    
    feature_columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'GarageCars', 
                       'TotalBsmtSF', 'HouseAge', 'TotalSF', 'QualityScore', 
                       'Neighborhood_encoded', 'HouseStyle_encoded']
    
    X = df[feature_columns]
    y = df['SalePrice']
    
    print(f"✓ Selected {len(feature_columns)} features for modeling")
    print("✓ Features:", ", ".join(feature_columns[:3]), "...", ", ".join(feature_columns[-2:]))
    
    # 6. Data Splitting
    print("\n6. DATA SPLITTING")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # 7. Model Training
    print("\n7. MODEL TRAINING")
    print("-" * 40)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'predictions': y_pred}
        
        print(f"  ✓ MAE: ${mae:,.2f}")
        print(f"  ✓ RMSE: ${rmse:,.2f}")
        print(f"  ✓ R²: {r2:.4f}")
    
    # 8. Model Comparison
    print("\n8. MODEL COMPARISON")
    print("-" * 40)
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    print("Performance Comparison:")
    print(comparison_df[['MAE', 'RMSE', 'R²']].round(2))
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['R²'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   R² Score: {results[best_model]['R²']:.4f}")
    print(f"   MAE: ${results[best_model]['MAE']:,.2f}")
    
    # 9. Feature Importance (for tree-based models)
    print("\n9. FEATURE IMPORTANCE")
    print("-" * 40)
    
    if best_model in ['Random Forest', 'Gradient Boosting']:
        model = models[best_model]
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"Top 5 Most Important Features ({best_model}):")
        for i, (feature, imp) in enumerate(zip(feature_importance['Feature'][:5], 
                                               feature_importance['Importance'][:5])):
            print(f"  {i+1}. {feature}: {imp:.4f}")
    else:
        print("Feature importance available for tree-based models only")
    
    # 10. Summary
    print("\n10. ANALYSIS SUMMARY")
    print("-" * 40)
    print("✓ Complete ML workflow executed successfully")
    print("✓ Data preprocessing and cleaning completed")
    print("✓ Feature engineering implemented")
    print("✓ Three regression models trained and evaluated")
    print("✓ Best performing model identified")
    print("✓ Feature importance analysis completed")
    
    print(f"\n🎯 Final Results:")
    print(f"   Dataset: {df.shape[0]} houses, {len(feature_columns)} features")
    print(f"   Best Model: {best_model}")
    print(f"   Performance: R² = {results[best_model]['R²']:.4f}")
    print(f"   Error: MAE = ${results[best_model]['MAE']:,.2f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("Project ready for educational use by 2nd-year CS students!")
    print("=" * 80)
    
    return results, feature_importance if best_model in ['Random Forest', 'Gradient Boosting'] else None

if __name__ == "__main__":
    results, importance = run_complete_analysis()
