import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import matplotlib.pyplot as plt

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

    # pass M as parameter and assert its value

    assert M == len(y) - 1, "Number of updates must be equal to length of y - 1."

    triangle_area = 0
    paralellogram_area = 0

    for i in range(M+1):
        triangle_area += (y[i]**2) / 2 # triangle area
        for j in range(i+1, M+1):
            paralellogram_area += y[i] * y[j] * (p)**(j - i) # I am using p as a error probability

    objective = (triangle_area + Z*paralellogram_area)/N
    avg_aoi = (triangle_area + paralellogram_area)/N
    avg_aois[Z] = avg_aoi # save the average AoI with Z as key

    return objective

M = 3 # Number of updates
N = 1  # Time interval
p_list =  np.linspace(0.0,0.5,20) # For M=3, Z=2 around p=0.4 we can see that number of updates drop
Z_list = [1,2,5,10]
AoIs = {}

y0 = np.ones(M+1) / (M+1) # 0, ..., M, M update epochs, M+1 intervals

lower_bounds = [0] * (M+1)
upper_bounds = [1] * (M+1)
bounds = Bounds(lower_bounds, upper_bounds)

A = np.ones((1, M+1))
lower_bound = [1] # the trick to have the sum EQUAL to 1 is that you set the lower limit to 1 and the upper limit also to 1
upper_bound = [1]
linear_constraint = LinearConstraint(A, lower_bound, upper_bound) 
#created 2 seperate graphs

line_styles = ['-', '--', '-.', ':']

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for i, z in enumerate(Z_list):
    result_values = []
    avg_aoi_values = []
    x_values = np.linspace(0.0, 0.5, len(p_list))
    for p in p_list:
        result = minimize(lambda ys: objective(ys, z, p, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
        result_values.append(result.fun)
        avg_aoi_values.append(AoIs[z])
    ax1.plot(x_values, result_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
    ax2.plot(x_values, avg_aoi_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')

ax1.set_xlabel("Probability (p)")
ax1.set_ylabel("Result")
ax1.set_xscale('log')  # Set the x-axis to logarithmic scale
ax1.legend()
ax1.set_title(f"Graph of the Minimal Penalty with update number of: {M}")

ax2.set_xlabel("Probability (p)")
ax2.set_ylabel("Avg AOI")
ax2.set_xscale('log')  # Set the x-axis to logarithmic scale
ax2.legend()
ax2.set_title(f"Graph of Avg AOI with update number of: {M}")

plt.tight_layout()
plt.show()
