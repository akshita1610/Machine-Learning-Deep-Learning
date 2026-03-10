"""
Script to run the Titanic notebook directly without Jupyter
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_notebook():
    """Execute the notebook and save results."""
    print("Running Titanic Survival Classification Notebook...")
    
    notebook_path = "Titanic_Survival_Classification.ipynb"
    output_path = "Titanic_Survival_Classification_Executed.ipynb"
    
    # Check if notebook exists
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found!")
        return False
    
    try:
        # Load the notebook
        print("Loading notebook...")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create execution preprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Execute the notebook
        print("Executing cells...")
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        
        # Save the executed notebook
        print("Saving executed notebook...")
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"Notebook executed successfully!")
        print(f"Results saved to: {output_path}")
        print("\nWhat was executed:")
        print("1. Data loading and preprocessing")
        print("2. Exploratory data analysis")
        print("3. Feature engineering")
        print("4. Model training (3 algorithms)")
        print("5. Model evaluation and comparison")
        print("6. Results and insights")
        
        return True
        
    except Exception as e:
        print(f"Error executing notebook: {e}")
        return False

if __name__ == "__main__":
    success = run_notebook()
    if success:
        print("\nTitanic project completed successfully!")
        print("Check the executed notebook for results and visualizations.")
    else:
        print("\nExecution failed. Check the error above.")
