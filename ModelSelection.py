# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset (use the first 1000 rows)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
df = pd.read_excel(url, header=1)

# Select the first 1000 rows
data_1000 = df.iloc[:1000, :]

# Features (exclude 'ID' and target 'default payment next month')
X = data_1000.drop(columns=['ID', 'default payment next month'])
y = data_1000['default payment next month']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a list of models to try
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Define a dictionary of hyperparameters for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10]
    }
}

# Function to perform model selection and fine-tuning
def tune_and_evaluate(models, param_grids, X_train, y_train, X_test, y_test):
    best_models = {}
    for name, model in models.items():
        print(f"Training and tuning {name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        # Evaluate the model on the test set
        y_pred = best_model.predict(X_test)
        print(f"Best Parameters for {name}: {grid_search.best_params_}")
        print(f"Accuracy for {name}: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Classification Report for {name}:\n {classification_report(y_test, y_pred)}\n")
    
    return best_models

# Perform model selection and fine-tuning
best_models = tune_and_evaluate(models, param_grids, X_train_scaled, y_train, X_test_scaled, y_test)

