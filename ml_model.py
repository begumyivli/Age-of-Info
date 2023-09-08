# dependencies
import pandas as pd
import optuna
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import matplotlib.pyplot as plt

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

y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

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

def objective(y:np.ndarray, Z:int, p:float, M:int, N:int, avg_aois:dict={}) -> float:
    """
    Objective function for the optimization problem.

    Parameters
    ----------
    y: numpy.ndarray
        vector of interupdate times.
    Z: int
        penalty for the paralellogram area.
    p: float
        error probability.
    M: int
        number of updates.
    N: int
        time interval.
    
    Optional
    --------
    avg_aois: dict
        dictionary to save the average AoI for each Z.

    Return
    ------
    objective: float
        value of the objective function.
    """

    assert M == len(y) - 1, "Number of updates must be equal to length of y - 1."

    triangle_area = 0
    paralellogram_area = 0

    for i in range(M+1):
        triangle_area += (y[i]**2) / 2 # triangle area
        for j in range(i+1, M+1):
            paralellogram_area += y[i] * y[j] * (p)**(j - i) # I am using p as a error probability

    objective = (triangle_area + Z*paralellogram_area)/N
    avg_aoi = (triangle_area + paralellogram_area)/N
    avg_aois[Z] = avg_aoi # save the average AoI with Z as key because for that probability we have 4 avg_aoi if we have 4 Z values
    
    return objective

M_values = [1,2,3,4,5,6] # Number of updates
N = 1  # Time interval
p = 1-accuracy
Z_list = [1,2,5,10]
AoIs = {}
fig1, ax1 = plt.subplots()
line_colors = ['blue','orange','purple','brown']

for i, z in enumerate(Z_list):
    result_values = []  
    avg_aoi_values = []
    interval_values = []
    for M in M_values:
        y0 = np.ones(M+1) / (M+1) # 0, ..., M, M update epochs, M+1 intervals

        lower_bounds = [0] * (M+1)
        upper_bounds = [1] * (M+1)
        bounds = Bounds(lower_bounds, upper_bounds)

        A = np.ones((1, M+1))
        lower_bound = [1] # the trick to have the sum EQUAL to 1 is that you set the lower limit to 1 and the upper limit also to 1
        upper_bound = [1]
        linear_constraint = LinearConstraint(A, lower_bound, upper_bound)

        result = minimize(lambda ys: objective(ys, z, p, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
        result_values.append(result.fun)  # Append the result value
        avg_aoi_values.append(AoIs[z])  # Calculate and append the avg_aoi value
                
        valid_interval = sum([1 for val in result.x if val > 1e-5])
        interval_values.append(valid_interval)
        num_of_updates = valid_interval-1

        """ cutting_idx = 0
        my_bool = False
        for j in range(len(interval_values)):
            if interval_values[j] <= M:
                print(interval_values[j])
                my_bool = True
                cutting_idx = j
                break """
           
    ax1.plot(M_values, result_values, label="Z = " + str(z), color=line_colors[i % len(line_colors)])
    # if we want to plot whole graphs just comment the last part from print and use last 3 line

x_tick_labels = [str(M) for M in M_values]

# Customize the tick labels on the x-axis
ax1.set_xticks(M_values)
ax1.set_xticklabels(x_tick_labels)

ax1.set_xlabel("M values")
ax1.set_ylabel("Minimal Penalty Value")
ax1.legend()
ax1.set_title(f"Graph of the Minimal Penalty")

plt.tight_layout()
plt.show()
fig1.savefig(f"outputs/plots/penalty_ml_model.png", bbox_inches='tight', pad_inches=0)