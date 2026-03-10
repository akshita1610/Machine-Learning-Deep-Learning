"""
Titanic Survival Prediction Demo
This script demonstrates how to use the trained model for making predictions.
"""

import pandas as pd
import numpy as np
import joblib

def load_model():
    """Load the trained model and preprocessing objects."""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, feature_columns

def prepare_sample_data():
    """Create sample passenger data for prediction."""
    # Load feature columns from the real data model
    _, _, feature_columns = load_model()
    
    # Create data with exactly the features the model expects
    sample_passengers = [
        {
            'Pclass': 1,
            'Sex': 1,  # female
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 100,
            'Has_Cabin': 1,
            'Embarked_C': 1,
            'Embarked_Q': 0,
            'Embarked_S': 0,
            'FamilySize': 1,
            'IsAlone': 1,
            'Title_encoded': 2,  # Mrs
            'AgeGroup_encoded': 2,  # Adult
            'FareGroup_encoded': 3,  # High
            'Title_Master': 0,
            'Title_Miss': 0,
            'Title_Mr': 0,
            'Title_Mrs': 1,
            'Title_Rare': 0,
            'Age_Adult': 1,
            'Age_Child': 0,
            'Age_Senior': 0,
            'Age_Teenager': 0,
            'Fare_High': 1,
            'Fare_Low': 0,
            'Fare_Medium-High': 0,
            'Fare_Medium-Low': 0
        },
        {
            'Pclass': 3,
            'Sex': 0,  # male
            'Age': 45,
            'SibSp': 1,
            'Parch': 2,
            'Fare': 20,
            'Has_Cabin': 0,
            'Embarked_C': 0,
            'Embarked_Q': 0,
            'Embarked_S': 1,
            'FamilySize': 4,
            'IsAlone': 0,
            'Title_encoded': 3,  # Mr
            'AgeGroup_encoded': 2,  # Adult
            'FareGroup_encoded': 1,  # Medium-Low
            'Title_Master': 0,
            'Title_Miss': 0,
            'Title_Mr': 1,
            'Title_Mrs': 0,
            'Title_Rare': 0,
            'Age_Adult': 1,
            'Age_Child': 0,
            'Age_Senior': 0,
            'Age_Teenager': 0,
            'Fare_High': 0,
            'Fare_Low': 0,
            'Fare_Medium-High': 0,
            'Fare_Medium-Low': 1
        }
    ]
    return pd.DataFrame(sample_passengers)

def predict_survival(passenger_data):
    """Make predictions on passenger data."""
    model, scaler, feature_columns = load_model()
    
    # Ensure we have all required columns
    for col in feature_columns:
        if col not in passenger_data.columns:
            passenger_data[col] = 0
    
    # Select only the features used in training
    X = passenger_data[feature_columns]
    
    # Scale numerical features
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities

def main():
    """Main demo function."""
    print("=== TITANIC SURVIVAL PREDICTION DEMO ===\n")
    
    # Load model and prepare data
    print("Loading trained model...")
    sample_data = prepare_sample_data()
    
    print("Sample passenger data:")
    print(sample_data[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']].to_string())
    print()
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = predict_survival(sample_data)
    
    # Display results
    passenger_types = ["First-class female", "Third-class male with family"]
    
    for i, (pred, prob, ptype) in enumerate(zip(predictions, probabilities, passenger_types)):
        survival = "Survived" if pred == 1 else "Did Not Survive"
        confidence = prob[pred] * 100
        
        print(f"\nPassenger {i+1} ({ptype}):")
        print(f"  Prediction: {survival}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Survival Probability: {prob[1]*100:.1f}%")
    
    print("\n=== DEMO COMPLETED ===")
    print("\nTo use this with real data:")
    print("1. Load your passenger data into a DataFrame")
    print("2. Ensure all required columns are present")
    print("3. Call predict_survival() function")
    print("4. Interpret the results")

if __name__ == "__main__":
    main()
