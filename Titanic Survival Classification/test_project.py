"""
Test script to verify Titanic project functionality
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

def test_imports():
    """Test if all required libraries can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_structure():
    """Test if project structure is correct."""
    print("\nTesting file structure...")
    required_dirs = ['data', 'models', 'notebooks', 'visualizations']
    required_files = ['Titanic_Survival_Classification.ipynb', 'README.md', 'requirements.txt']
    
    all_good = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory '{dir_name}' exists")
        else:
            print(f"❌ Directory '{dir_name}' missing")
            all_good = False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File '{file_name}' exists")
        else:
            print(f"❌ File '{file_name}' missing")
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
                print(f"✅ {file_path} loads successfully")
            except Exception as e:
                print(f"❌ {file_path} loading error: {e}")
                all_good = False
        else:
            print(f"❌ {file_path} missing")
            all_good = False
    
    return all_good

def test_demo_script():
    """Test if demo script runs without errors."""
    print("\nTesting demo script...")
    try:
        # Import and run demo functions
        import demo
        model, scaler, feature_columns = demo.load_model()
        sample_data = demo.prepare_sample_data()
        predictions, probabilities = demo.predict_survival(sample_data)
        
        print("✅ Demo script runs successfully")
        print(f"✅ Made {len(predictions)} predictions")
        return True
    except Exception as e:
        print(f"❌ Demo script error: {e}")
        return False

def test_basic_ml_workflow():
    """Test basic ML workflow with sample data."""
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
        
        print(f"✅ ML workflow successful (accuracy: {accuracy:.3f})")
        return True
    except Exception as e:
        print(f"❌ ML workflow error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TITANIC PROJECT FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Library Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Model Files", test_model_files),
        ("Demo Script", test_demo_script),
        ("Basic ML Workflow", test_basic_ml_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
