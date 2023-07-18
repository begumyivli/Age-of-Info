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
        success probability.
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
            paralellogram_area += y[i] * y[j] * (1 - p)**(j - i) # paralelogram area Zyi eklemen gerek??

    objective = (triangle_area + Z*paralellogram_area)/N
    avg_aoi = (triangle_area + paralellogram_area)/N
    avg_aois[Z] = avg_aoi # save the average AoI with Z as key

    return objective

M = 3 # Number of updates
N = 1  # Time interval
p_list =  np.linspace(0.5,1,20) #[0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.63, 0.65, 0.68, 0.7, 0.73, 0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.92, 0.95, 0.97, 1]
print(p_list)
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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

x_values = np.linspace(0.5, 1, len(p_list))

line_styles = ['-', '--', '-.', ':']

#x_values = np.arange(len(p_list)) bunu kullanma equally spaced sebep oluyor

for i, z in enumerate(Z_list):
    result_values = []  # Store the result values for each p in p_list
    avg_aoi_values = []  # Store the avg_aoi values for each p in p_list
    update_values = []

    for j, p in enumerate(p_list):
        result = minimize(lambda ys: objective(ys, z, p, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
        result_values.append(result.fun)  # Append the result value
        avg_aoi_values.append(AoIs[z])  # Calculate and append the avg_aoi value
        
        num_updates = sum([1 for val in result.x if val > 1e-5])
        update_values.append(num_updates)

        print(f"Z: {z}")
        print(f"p: {p}")
        print(f"Average Age of Information with penalty: {result.fun:.5f}")
        print(f"Average Age of Information without penalty: {AoIs[z]:.5f}")
        print(f"Optimal interupdate times: {result.x} \n") 

    ax1.plot(x_values, result_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
    ax2.plot(x_values, avg_aoi_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
    ax3.plot(x_values, update_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')

# Customize the tick labels on the x-axis
ax1.set_xticks(x_values)
ax1.set_xticklabels([round(x, 2) for x in x_values])
ax2.set_xticks(x_values)
ax2.set_xticklabels([round(x, 2) for x in x_values])
ax3.set_xticks(x_values)
ax3.set_xticklabels([round(x, 2) for x in x_values])

ax1.set_xlabel("Probability (p)")
ax1.set_ylabel("Result")
ax1.legend()
ax1.set_title(f"Graph of the Minimal Penalty with update number of: {M}")

ax2.set_xlabel("Probability (p)")
ax2.set_ylabel("Avg AOI")
ax2.legend()
ax2.set_title(f"Graph of Avg AOI with update number of: {M}")

ax3.set_xlabel("Probability (p)")
ax3.set_ylabel("Number of Updates")
ax3.legend()
ax3.set_title("Graph of Number of Updates")

plt.tight_layout()
plt.show()

""" for z in Z_list:
    result_values = []  # Store the result values for each p in p_list
    avg_aoi_values = []  # Store the avg_aoi values for each p in p_list
    for p in p_list:
        result = minimize(lambda y0: objective(y0, z), y0, bounds=bounds, constraints=linear_constraint)
        result_values.append(result.fun)  # Append the result value
        avg_aoi_values.append(avg_aoi)  # Append the avg_aoi value

    plt.plot(p_list, result_values, label="Z = " + str(z))
    plt.plot(p_list, avg_aoi_values, label="Avg AOI, Z = " + str(z), linestyle="--")

plt.xlabel("Probability (p)")
plt.ylabel("Result / Avg AOI")
plt.legend()
plt.title("Graph of the Minimal Penalty")

plt.show() """


# fun: The optimal value of the objective function obtained after the optimization process
# x: The optimal solution, i.e., the values of the variables that minimize the objective function
# nit: The number of iterations performed during the optimization process
# jac: The Jacobian of the objective function at the optimal solution. This represents the gradient of the objective function with respect to each variable
# nfev: The number of evaluations of the objective function
# njev: The number of evaluations of the Jacobian (gradient) of the objective function