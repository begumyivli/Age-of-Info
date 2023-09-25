import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint
from joblib import dump, load
from sklearn.metrics import accuracy_score
import matplotlib

# Comment these if you don't want to output pgfs
#matplotlib.use('pgf')
#matplotlib.rcParams.update({
#    'pgf.texsystem': 'pdflatex',
#    'font.family': 'serif',
#    'font.size': 14,
#    'text.usetex': True,
#    'pgf.rcfonts': False,
#})

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

accuracy_scores = {}
with open('accuracy_scores.txt', 'r') as file:
    for line in file:
        model, score = line.strip().split(': ')
        accuracy_scores[model] = float(score)

# Find the lowest and highest scores
lowest_score_model = min(accuracy_scores, key=accuracy_scores.get)
highest_score_model = max(accuracy_scores, key=accuracy_scores.get)

AoIs = {}
fig1, ax1 = plt.subplots()
M_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Number of updates
N = 1  # Time interval
p_values = [1 - accuracy_scores[lowest_score_model], 1 - accuracy_scores[highest_score_model]]  # Two p values
Z_list = [1, 2, 5, 10, 15]  # List of Z values
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
marker_styles = ['o', 'd', 's', '^', 'v']

for p in p_values:
    base = None
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
            if z==10:
                print(p)
                print(num_of_updates)

            """ cutting_idx = 0
            my_bool = False
            for j in range(len(interval_values)):
                if interval_values[j] <= M:
                    print(interval_values[j])
                    my_bool = True
                    cutting_idx = j
                    break """
        
        result_values = np.array(result_values)
        
        if i == 0:
            base = np.array(result_values)
            continue
        ax1.plot(
            M_values,
            result_values/base,
            label=f"p = {p:.3f}, Z = {z}",
            color=line_colors[i % len(line_colors)],
            linestyle='-' if p == 1-accuracy_scores[highest_score_model] else '--',
            marker = 'd' if p == 1-accuracy_scores[highest_score_model] else 'o',
            markevery=0.2
        )
        # if we want to plot whole graphs just comment the last part from print and use last 3 line

x_tick_labels = [str(M) for M in M_values]

# Customize the tick labels on the x-axis
ax1.set_xticks(M_values)
ax1.set_xticklabels(x_tick_labels)
ax1.grid(True, alpha=0.5)

ax1.set_xlabel("Number of updates ($M$)")
ax1.set_ylabel("Minimum penalty over baseline")
ax1.legend()

plt.show()
#fig1.savefig(f"outputs/plots/penalty_ml_model.pgf", bbox_inches='tight', pad_inches=0)