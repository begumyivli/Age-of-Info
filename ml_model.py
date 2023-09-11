# dependencies
import joblib
import pandas as pd
import optuna
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump, load

dataset = pd.read_csv('TUANDROMD.csv')
df = dataset.dropna() # cleaned dataset

X = df.drop(columns=['Label'], axis=1)  # Features
y = df['Label']

# Encode the categorical target variable using Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = 1 - y_encoded

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-3, 1)
    gamma = trial.suggest_loguniform('gamma', 1e-1, 1e1)

    # Create the SVM classifier with RBF kernel using suggested hyperparameters
    svm_classifier = SVC(kernel='rbf', C=C, gamma=gamma)

    # Use stratified k-fold cross-validation for better validation performance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Calculate cross-validated accuracy with stratification
    cv_accuracy = cross_val_score(svm_classifier, X_train, y_train, cv=cv, scoring='accuracy').mean()

    print(cv_accuracy)
    return cv_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100) # adjust n_trials

# Get the best hyperparameters
n_trials = len(study.trials)
best_params = study.best_params

best_C = best_params['C']
best_gamma = best_params['gamma']

svm_classifier = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
svm_classifier = svm_classifier.fit(X_train, y_train)

dump(svm_classifier, 'svm_classifier.joblib')

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

import pickle

with open("accuracy.pkl", "wb") as file:
    pickle.dump(accuracy, file)

file_path = "model_evaluation.txt"

with open(file_path, 'a') as file:
    # Write the best hyperparameters
    file.write("Best Hyperparameters:\n")
    file.write(f"C: {best_C}\n")
    file.write(f"Gamma: {best_gamma}\n")
    file.write(f"n_trials: {n_trials}\n")
    
    # Write the evaluation scores
    file.write("\nEvaluation Scores:\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1-score: {f1}\n")
    file.write("\n")

print(f"Model evaluation information saved to {file_path}")
