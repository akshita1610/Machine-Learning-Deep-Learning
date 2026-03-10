"""
Simple test script to verify Titanic project functionality
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

def test_core_imports():
    """Test core ML libraries."""
    print("Testing core imports...")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        print("PASS: Core imports successful")
        return True
    except ImportError as e:
        print(f"FAIL: Import error: {e}")
        return False

def test_file_structure():
    """Test project structure."""
    print("\nTesting file structure...")
    required_files = [
        'Titanic_Survival_Classification.ipynb', 
        'README.md', 
        'requirements.txt',
        'demo.py'
    ]
    required_dirs = ['data', 'models', 'notebooks', 'visualizations']
    
    all_good = True
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"PASS: {file_name} exists")
        else:
            print(f"FAIL: {file_name} missing")
            all_good = False
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"PASS: {dir_name}/ exists")
        else:
            print(f"FAIL: {dir_name}/ missing")
            all_good = False
    
    return all_good

def test_model_files():
    """Test if model files exist and can be loaded."""
    print("\nTesting model files...")
    model_files = ['models/best_model.pkl', 'models/scaler.pkl', 'models/feature_columns.pkl']
    
    all_good = True
    for file_path in model_files:
        if os.path.exists(file_path):
            try:
                obj = joblib.load(file_path)
                print(f"PASS: {file_path} loads successfully")
            except Exception as e:
                print(f"FAIL: {file_path} loading error: {e}")
                all_good = False
        else:
            print(f"FAIL: {file_path} missing")
            all_good = False
    
    return all_good

def test_basic_workflow():
    """Test basic ML workflow."""
    print("\nTesting basic ML workflow...")
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"PASS: ML workflow successful (accuracy: {accuracy:.3f})")
        return True
    except Exception as e:
        print(f"FAIL: ML workflow error: {e}")
        return False

def test_demo_functionality():
    """Test demo script functionality."""
    print("\nTesting demo functionality...")
    try:
        # Load model files
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        
        # Create test data
        test_data = pd.DataFrame({
            'Pclass': [1, 3],
            'Sex': [1, 0],
            'Age': [25, 45],
            'SibSp': [0, 1],
            'Parch': [0, 2],
            'Fare': [100, 20],
            'FamilySize': [1, 4],
            'IsAlone': [1, 0],
            'Embarked_C': [1, 0],
            'Embarked_Q': [0, 0],
            'Embarked_S': [0, 1]
        })
        
        # Make prediction
        X = test_data[feature_columns]
        numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        X[numerical_cols] = scaler.transform(X[numerical_cols])
        predictions = model.predict(X)
        
        print(f"PASS: Demo functionality works (made {len(predictions)} predictions)")
        return True
    except Exception as e:
        print(f"FAIL: Demo functionality error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TITANIC PROJECT FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("File Structure", test_file_structure),
        ("Model Files", test_model_files),
        ("Basic Workflow", test_basic_workflow),
        ("Demo Functionality", test_demo_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAIL: {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Project is working correctly.")
    else:
        print("WARNING: Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
