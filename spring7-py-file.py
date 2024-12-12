import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
try:
    data = pd.read_csv('datasets/users_behavior.csv')
except:
    data = pd.read_csv('https://practicum-content.s3.us-west-1.amazonaws.com/datasets/users_behavior.csv')

data

# Separate features and target
X = data.drop('is_ultra', axis=1)
y = data['is_ultra']

# Split data: 60% training, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Output the shapes of the splits
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_valid.shape, y_valid.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models and hyperparameter grids
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
    },
    'SVM': {
        'model': SVC(random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
}

# Perform GridSearchCV for each model
results = []
for model_name, model_details in models.items():
    grid = GridSearchCV(
        estimator=model_details['model'],
        param_grid=model_details['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({
        'Model': model_name,
        'Best Params': grid.best_params_,
        'Accuracy': accuracy
    })
    print(f"Model: {model_name}")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Display results summary
results_df = pd.DataFrame(results)
print(results_df)