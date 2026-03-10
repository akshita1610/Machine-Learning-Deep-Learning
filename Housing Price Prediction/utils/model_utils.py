"""
Model Training and Evaluation Utilities

This module provides functions for training, evaluating, and saving
machine learning models for housing price prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A class for training and evaluating regression models.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Initialize the regression models to be compared.
        
        Returns:
        --------
        dict
            Dictionary of initialized models
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.models = models
        print(f"Initialized {len(models)} models for comparison")
        return models
    
    def train_models(self, X_train, y_train, X_test, y_test, scaler=None):
        """
        Train all models and evaluate their performance.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training target
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            Test target
        scaler : StandardScaler, optional
            Fitted scaler for linear regression
            
        Returns:
        --------
        dict
            Dictionary containing model results
        """
        print("Training models...")
        results = {}
        predictions = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'Linear Regression' and scaler is not None:
                # Linear Regression benefits from scaled features
                model.fit(scaler.transform(X_train), y_train)
                y_pred = model.predict(scaler.transform(X_test))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Store predictions
            predictions[name] = y_pred
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"MAE: ${mae:,.2f}")
            print(f"RMSE: ${rmse:,.2f}")
            print(f"R²: {r2:.4f}")
        
        self.results = results
        self._find_best_model()
        
        return results
    
    def _find_best_model(self):
        """
        Find the best performing model based on R² score.
        """
        if not self.results:
            return
            
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['R²'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"R² Score: {self.results[best_model_name]['R²']:.4f}")
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importance = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return feature_importance_df
        else:
            # For Linear Regression, return coefficients
            if hasattr(self.best_model, 'coef_'):
                coefficients = self.best_model.coef_
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                return coef_df
        
        return None
    
    def plot_model_comparison(self):
        """
        Create visualization comparing model performance.
        """
        if not self.results:
            print("No results to plot. Train models first.")
            return
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metric comparison
        metrics = ['MAE', 'RMSE', 'R²']
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            results_df[metric].sort_values().plot(kind='bar', ax=axes[row, col])
            axes[row, col].set_title(f'{metric} Comparison')
            axes[row, col].set_ylabel(metric)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        # Remove the empty subplot
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_test):
        """
        Plot predictions vs actual values for the best model.
        
        Parameters:
        -----------
        y_test : pd.Series or np.array
            Actual test values
        """
        if not self.best_model_name:
            print("No best model found. Train models first.")
            return
        
        best_predictions = self.results[self.best_model_name]['predictions']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, best_predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'{self.best_model_name}: Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_model(self, model_path, scaler_path=None, feature_columns_path=None, 
                   label_encoders_path=None):
        """
        Save the best model and preprocessing objects.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        scaler_path : str, optional
            Path to save the scaler
        feature_columns_path : str, optional
            Path to save feature columns
        label_encoders_path : str, optional
            Path to save label encoders
        """
        if not self.best_model:
            print("No model to save. Train models first.")
            return
        
        # Save model
        joblib.dump(self.best_model, model_path)
        print(f"Best model ({self.best_model_name}) saved to: {model_path}")
        
        # Save additional objects if paths provided
        if scaler_path and hasattr(self, 'scaler'):
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to: {scaler_path}")
        
        if feature_columns_path and hasattr(self, 'feature_columns'):
            joblib.dump(self.feature_columns, feature_columns_path)
            print(f"Feature columns saved to: {feature_columns_path}")
        
        if label_encoders_path and hasattr(self, 'label_encoders'):
            joblib.dump(self.label_encoders, label_encoders_path)
            print(f"Label encoders saved to: {label_encoders_path}")
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        object
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            print(f"Model loaded successfully from: {model_path}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def cross_validate_model(self, X, y, model_name=None, cv=5):
        """
        Perform cross-validation on a model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features
        y : pd.Series or np.array
            Target
        model_name : str, optional
            Name of the model to validate
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        results = {
            'CV Scores': cv_scores,
            'Mean CV Score': cv_scores.mean(),
            'Std CV Score': cv_scores.std()
        }
        
        print(f"Cross-validation results for {model_name}:")
        print(f"Mean R²: {results['Mean CV Score']:.4f} (+/- {results['Std CV Score']:.4f})")
        
        return results


def create_comparison_table(results):
    """
    Create a formatted comparison table of model results.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
        
    Returns:
    --------
    pd.DataFrame
        Formatted comparison table
    """
    # Create dataframe from results
    comparison_df = pd.DataFrame(results).T
    
    # Format numeric columns
    for col in ['MAE', 'RMSE']:
        comparison_df[col] = comparison_df[col].apply(lambda x: f"${x:,.2f}")
    
    comparison_df['R²'] = comparison_df['R²'].apply(lambda x: f"{x:.4f}")
    
    return comparison_df


def plot_feature_importance(importance_df, title="Feature Importance", top_n=10):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    title : str
        Plot title
    top_n : int
        Number of top features to show
    """
    if importance_df is None:
        print("No importance data to plot.")
        return
    
    # Get top features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    
    if 'Importance' in top_features.columns:
        # Tree-based models
        sns.barplot(data=top_features, x='Importance', y='Feature')
        plt.xlabel('Importance')
    else:
        # Linear regression coefficients
        sns.barplot(data=top_features, x='Coefficient', y='Feature')
        plt.xlabel('Coefficient')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from data_loader import HousingDataLoader, load_sample_data
    
    # Load and prepare data
    loader = HousingDataLoader()
    df = load_sample_data()
    df = loader.clean_data(df)
    df = loader.encode_categorical(df)
    df = loader.create_features(df)
    X, y = loader.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    trainer = ModelTrainer()
    trainer.initialize_models()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Plot results
    trainer.plot_model_comparison()
    trainer.plot_predictions(y_test)
    
    # Get feature importance
    importance_df = trainer.get_feature_importance(X.columns)
    if importance_df is not None:
        plot_feature_importance(importance_df, f"Feature Importance - {trainer.best_model_name}")
    
    # Create comparison table
    comparison_table = create_comparison_table(results)
    print("\nModel Comparison Table:")
    print(comparison_table)
