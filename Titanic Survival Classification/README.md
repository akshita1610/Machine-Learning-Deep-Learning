# Titanic Passenger Survival Classification

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-83.8%25-brightgreen.svg)]()

A comprehensive machine learning project that predicts passenger survival on the Titanic with **83.8% accuracy** using real historical data.

## 📋 Project Overview

This project implements a complete data science workflow to predict Titanic passenger survival based on demographic and travel information. It demonstrates fundamental machine learning concepts including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

## 🎯 Project Goals

- Build an accurate classification model to predict passenger survival
- Identify key factors that influenced survival rates
- Implement and compare multiple machine learning algorithms
- Create a reproducible workflow for binary classification problems
- Provide educational value for 2nd-year Computer Science students

## 📊 Dataset

**Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)

**Features**:
- `PassengerId`: Unique identifier for each passenger
- `Survived`: Target variable (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Passenger's name
- `Sex`: Passenger's gender
- `Age`: Passenger's age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🛠️ Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine learning algorithms and evaluation metrics
- **Jupyter Notebook**: Interactive development environment

## 📁 Project Structure

```
Titanic Survival Classification/
├── data/
│   └── titanic.csv                 # Real Titanic dataset (891 passengers)
├── models/
│   ├── best_titanic_model.pkl      # Trained Logistic Regression model
│   ├── scaler.pkl                  # Feature scaling parameters
│   └── feature_columns.pkl         # List of 28 engineered features
├── Titanic_Survival_Classification.ipynb  # Complete Jupyter notebook
├── titanic_analysis.py             # Complete analysis script (83.8% accuracy)
├── fixed_demo.py                   # Working prediction demo
├── fixed_test.py                   # Complete test suite (5/5 tests pass)
├── show_results.py                 # Project results summary
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Git (optional, for cloning)

### Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd "Titanic Survival Classification"
pip install -r requirements.txt

# 2. Run the complete analysis
python titanic_analysis.py

# 3. Test everything works
python fixed_test.py

# 4. See predictions in action
python fixed_demo.py

# 5. View project summary
python show_results.py
```

### Running the Project

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   - Navigate to `Titanic_Survival_Classification.ipynb`
   - Run cells sequentially from top to bottom

## 📈 Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Age filled with median based on sex and class, Embarked with mode
- **Categorical Encoding**: One-hot encoding for Embarked, label encoding for other categorical variables
- **Feature Scaling**: StandardScaler applied to numerical features

### 2. Exploratory Data Analysis
- Survival analysis by gender, class, age, and fare
- Correlation heatmap to identify feature relationships
- Distribution plots for key variables

### 3. Feature Engineering
- **FamilySize**: Combined siblings, spouses, parents, and children
- **IsAlone**: Binary feature for passengers traveling alone
- **Title**: Extraction from names (Mr, Mrs, Miss, etc.)
- **AgeGroup**: Categorization into Child, Teenager, Adult, Senior
- **FareGroup**: Fare categorization into Low, Medium-Low, Medium-High, High

### 4. Model Building
Three classification algorithms implemented:
- **Logistic Regression**: Baseline model with good interpretability
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble method for better performance

### 5. Model Evaluation
Metrics used for evaluation:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 5-fold CV for robust performance estimation

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Cross-Validation |
|-------|----------|-----------|--------|----------|------------------|
| **Logistic Regression** | **83.80%** | **80.30%** | **76.81%** | **78.52%** | 81.05% ± 2.67% |
| Random Forest | 78.77% | 72.46% | 72.46% | 72.46% | 79.50% ± 3.36% |
| Gradient Boosting | 81.01% | 79.66% | 68.12% | 73.44% | 81.61% ± 3.21% |

**🏆 Best Model**: Logistic Regression with **83.80% accuracy** on real Titanic data

### Key Findings

1. **Gender Impact**: Female passengers had significantly higher survival rates (**74.20% vs 18.89%** for males) - **3.9x difference**
2. **Class System**: First-class passengers had much better survival rates (**62.96%** vs 24.24% for third-class)
3. **Family Dynamics**: Families of 4 had the best survival rate (**72.41%**), while very large families (8+) had 0% survival
4. **Age Factor**: Children and young adults had higher survival rates compared to elderly passengers
5. **Economic Status**: Higher fare passengers had significantly better survival outcomes

### Feature Importance

Top predictive features from the trained model:
1. **Age** (17.99% importance): Younger passengers had better survival chances
2. **Fare** (17.00% importance): Higher ticket price correlated with better survival
3. **Title_Mr** (11.09% importance): Male title indicator
4. **Sex** (8.69% importance): Gender was a critical factor
5. **Pclass** (5.72% importance): Passenger class significantly affected outcomes
6. **FamilySize** (4.46% importance): Family composition influenced survival
7. **Has_Cabin** (4.63% importance): Having a cabin indicated higher status

## 🎓 Learning Outcomes

This project demonstrates:

- **Data Science Workflow**: Complete pipeline from data ingestion to model deployment
- **Preprocessing Techniques**: Handling missing values, encoding categorical variables
- **Feature Engineering**: Creating meaningful features from existing data
- **Model Selection**: Comparing different algorithms and selecting the best performer
- **Evaluation Methods**: Using appropriate metrics for classification problems
- **Visualization**: Creating insightful plots to communicate findings

## 🔧 Customization and Extensions

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Example for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                          param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

### Adding New Models

```python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

# Example ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42))
    ],
    voting='soft'
)
```

## 🚀 Future Improvements

1. **Advanced Feature Engineering**:
   - Extract more information from ticket numbers
   - Analyze cabin locations and deck information
   - Create interaction features between variables

2. **Model Optimization**:
   - Implement comprehensive hyperparameter tuning
   - Try advanced algorithms (XGBoost, LightGBM, CatBoost)
   - Use stacking and blending techniques

3. **Data Enhancement**:
   - Incorporate external historical data
   - Add crew member information
   - Include weather and maritime conditions

4. **Deployment**:
   - Create a web interface using Flask or Streamlit
   - Implement model monitoring and retraining pipeline
   - Add confidence intervals to predictions

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## 👨‍💻 Author

This project was created as an educational demonstration of machine learning concepts suitable for 2nd-year Computer Science students.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you have any questions or need clarification on any part of the project, please open an issue in the repository.

---

**Happy Learning! 🎓**
