"""
Simple Jupyter Notebook Launcher

This script provides multiple ways to access the Jupyter notebook
if the direct browser preview doesn't work.
"""

import os
import subprocess
import webbrowser
import time
import sys

def check_jupyter_running():
    """Check if Jupyter is running on common ports."""
    import socket
    ports = [8888, 8889, 8890]
    
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return port
        except:
            continue
    return None

def start_jupyter_server():
    """Start Jupyter server with different configurations."""
    print("Starting Jupyter server...")
    
    # Try different configurations
    configs = [
        ['python', '-m', 'notebook', '--port=8888', '--no-browser', '--ip=127.0.0.1'],
        ['python', '-m', 'notebook', '--port=8889', '--no-browser', '--ip=127.0.0.1'],
        ['python', '-m', 'notebook', '--port=8890', '--no-browser', '--ip=127.0.0.1'],
    ]
    
    for config in configs:
        port = config[2].split('=')[1]
        print(f"Trying port {port}...")
        
        try:
            # Start the server
            process = subprocess.Popen(config, cwd=os.getcwd(), 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if it's working
            if check_jupyter_running():
                print(f"Jupyter server started successfully on port {port}")
                return port, process
                
        except Exception as e:
            print(f"Failed to start on port {port}: {e}")
            continue
    
    return None, None

def open_browser_manually():
    """Provide manual instructions to open the notebook."""
    print("\n" + "="*60)
    print("MANUAL JUPYTER ACCESS INSTRUCTIONS")
    print("="*60)
    print()
    print("Option 1: Open Jupyter directly")
    print("  1. Open Command Prompt or PowerShell")
    print("  2. Navigate to the project folder:")
    print('     cd "C:\\Users\\akshi\\OneDrive\\Desktop\\Projects\\Housing Price Prediction"')
    print("  3. Run: python -m notebook")
    print("  4. Copy the URL that appears (usually http://localhost:8888)")
    print("  5. Paste it in your browser")
    print()
    print("Option 2: Use VS Code")
    print("  1. Open VS Code")
    print("  2. Open the project folder")
    print("  3. Open: notebooks/housing_price_prediction.ipynb")
    print("  4. Run cells using the VS Code Jupyter extension")
    print()
    print("Option 3: Direct file access")
    print("  1. Navigate to: notebooks/housing_price_prediction.ipynb")
    print("  2. Right-click -> Open with -> Jupyter Notebook")
    print()
    print("="*60)

def create_simple_notebook_runner():
    """Create a simple script to run the notebook directly."""
    runner_code = '''
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

print("\\nResults:")
for name, metrics in results.items():
    print(f"{name}: R² = {metrics['R²']:.4f}")

print("\\n=== Direct Run Completed Successfully! ===")
'''
    
    with open('run_direct.py', 'w') as f:
        f.write(runner_code)
    
    print("Created run_direct.py for direct execution")

def main():
    """Main function to provide access options."""
    print("Housing Price Prediction - Notebook Access")
    print("="*50)
    
    # Check if Jupyter is already running
    running_port = check_jupyter_running()
    if running_port:
        print(f"Jupyter is already running on port {running_port}")
        print(f"Access it at: http://localhost:{running_port}")
        return
    
    # Try to start Jupyter
    port, process = start_jupyter_server()
    
    if port:
        print(f"\nSUCCESS! Jupyter is running on port {port}")
        print(f"Access it at: http://localhost:{port}")
        print("\nNavigate to: notebooks/housing_price_prediction.ipynb")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{port}')
            print("Browser opened automatically!")
        except:
            print("Could not open browser automatically. Please open manually.")
    else:
        print("\nCould not start Jupyter server automatically.")
        print("\nHere are your options:")
        
        # Create direct runner
        create_simple_notebook_runner()
        
        # Provide manual instructions
        open_browser_manually()
        
        print("\nQuick test - Run directly:")
        print("python run_direct.py")

if __name__ == "__main__":
    main()
