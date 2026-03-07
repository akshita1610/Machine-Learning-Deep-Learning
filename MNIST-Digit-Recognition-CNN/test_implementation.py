"""
Quick test script to verify the MNIST CNN implementation works correctly.
This script tests the core functionality without running full training.
"""

import numpy as np
import sys
import os

# Add current directory to path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"[PASS] TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"[FAIL] TensorFlow import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"[PASS] Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"[FAIL] Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn
        print(f"[PASS] Seaborn version: {seaborn.__version__}")
    except ImportError as e:
        print(f"[FAIL] Seaborn import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"[PASS] Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"[FAIL] Scikit-learn import failed: {e}")
        return False
    
    return True

def test_mnist_data():
    """Test that MNIST data can be loaded."""
    print("\nTesting MNIST data loading...")
    
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        print(f"[PASS] MNIST data loaded successfully")
        print(f"  Training set shape: {x_train.shape}")
        print(f"  Test set shape: {x_test.shape}")
        print(f"  Labels shape: {y_train.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] MNIST data loading failed: {e}")
        return False

def test_model_creation():
    """Test that the model can be created without training."""
    print("\nTesting model creation...")
    
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, models
        
        # Create a simple model for testing
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("[PASS] Model created successfully")
        print(f"  Model parameters: {model.count_params():,}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False

def test_our_class():
    """Test that our MNISTDigitRecognizer class can be imported and instantiated."""
    print("\nTesting MNISTDigitRecognizer class...")
    
    try:
        # Try to import without running the main function
        import importlib.util
        spec = importlib.util.spec_from_file_location("mnist_cnn", "mnist_cnn.py")
        mnist_module = importlib.util.module_from_spec(spec)
        
        # Execute the module but skip the main() call
        old_main = None
        if hasattr(mnist_module, 'main'):
            old_main = mnist_module.main
            mnist_module.main = lambda: None  # Replace main with no-op
        
        spec.loader.exec_module(mnist_module)
        
        # Restore main if it existed
        if old_main:
            mnist_module.main = old_main
        
        # Test class instantiation
        recognizer = mnist_module.MNISTDigitRecognizer()
        print("[PASS] MNISTDigitRecognizer class instantiated successfully")
        
        # Test data loading
        recognizer.load_and_preprocess_data(validation_split=0.1)
        print("[PASS] Data loading and preprocessing successful")
        
        # Test model building
        recognizer.build_model()
        print("[PASS] Model building successful")
        
        return True
    except Exception as e:
        print(f"[FAIL] MNISTDigitRecognizer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MNIST CNN Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("MNIST Data Loading", test_mnist_data),
        ("Model Creation", test_model_creation),
        ("Our Implementation", test_our_class),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[PASS]" if result else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! The implementation should work correctly.")
        print("\nYou can now run the full training with:")
        print("python mnist_cnn.py")
    else:
        print("[WARNING] Some tests failed. Please check the error messages above.")
        print("You may need to install missing packages or fix configuration issues.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
