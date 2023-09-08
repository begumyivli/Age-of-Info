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
    avg_aois[Z] = avg_aoi # save the average AoI with Z as key because for that probability we have 4 avg_aoi if we have 4 Z values
    

    return objective

M = 3 # Number of updates
N = 1  # Time interval
p_list =  np.linspace(0.0,0.5,20) # For M=3, Z=2 around p=0.4 we can see that number of updates drop
#p_list = [0.10]
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

# creating 3 separate graphs thanks to different axes
#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
#fig3, ax3 = plt.subplots()

x_values = np.linspace(0.0, 0.5, len(p_list))

#line_styles = ['-', '--', '-.', ':']
line_colors = ['blue','orange','purple','brown']

#x_values = np.arange(len(p_list)) bunu kullanma equally spaced sebep oluyor
#file1 = open("results M:"+str(M)+".txt",'w')
file1 = open("outputs/update/update:"+str(M)+"equal prob.txt",'w')

for i, z in enumerate(Z_list):
    result_values = []  # Store the result values for each p in p_list
    avg_aoi_values = []  # Store the avg_aoi values for each p in p_list
    interval_values = []

    for j, p in enumerate(p_list):
        result = minimize(lambda ys: objective(ys, z, p, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
        result_values.append(result.fun)  # Append the result value
        avg_aoi_values.append(AoIs[z])  # Calculate and append the avg_aoi value
        
        valid_interval = sum([1 for val in result.x if val > 1e-5])
        interval_values.append(valid_interval)
        num_of_updates = valid_interval-1

        file1.write(f"Z:{z} ")
        file1.write(f"p:{p} ")
        file1.write(f"Number of Updates:{num_of_updates} ")
        file1.write(f"Average Age of Information with penalty:{result.fun:.5f} ")
        file1.write(f"Average Age of Information without penalty:{AoIs[z]:.5f} ")
        file1.write(f"Optimal interupdate times: {result.x} \n") 
    file1.write("\n")

    cutting_idx = 0
    my_bool = False
    for j in range(len(interval_values)):
        if interval_values[j] <= M:
            my_bool = True
            cutting_idx = j
            break
    
    if my_bool:
        ax1.plot(x_values[:cutting_idx+1], result_values[:cutting_idx+1], label="Z = " + str(z), color=line_colors[i % len(line_colors)])
        ax1.plot(x_values[cutting_idx:], result_values[cutting_idx:], linestyle='--', color=line_colors[i % len(line_colors)])
        ax2.plot(x_values[:cutting_idx+1], avg_aoi_values[:cutting_idx+1], label="Z = " + str(z), color=line_colors[i % len(line_colors)])
        ax2.plot(x_values[cutting_idx:], avg_aoi_values[cutting_idx:], linestyle='--', color=line_colors[i % len(line_colors)])
    else:
        ax1.plot(x_values, result_values, label="Z = " + str(z), color=line_colors[i % len(line_colors)])
        ax2.plot(x_values, avg_aoi_values, label="Z = " + str(z), color=line_colors[i % len(line_colors)])
    #ax3.plot(x_values, interval_values, label="Z = " + str(z), color=line_colors[i % len(line_colors)])
    # if we want to plot whole graphs just comment the last part from print and use last 3 line

# remember to close the file after writing
file1.close()

x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
x_tick_labels = [str(x) for x in x_ticks]

# Customize the tick labels on the x-axis
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_tick_labels)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_tick_labels)
""" ax3.set_xticks(x_ticks)
ax3.set_xticklabels(x_tick_labels) """

ax1.set_xlabel("Probability (p)")
ax1.set_ylabel("Minimal Penalty Value")
ax1.legend()
ax1.set_title(f"Graph of the Minimal Penalty with update number of: {M}")

ax2.set_xlabel("Probability (p)")
ax2.set_ylabel("Avg AOI")
ax2.legend()
ax2.set_title(f"Graph of Avg AOI with update number of: {M}")

""" ax3.set_xlabel("Probability (p)")
ax3.set_ylabel("Number of Valid Intervals\n(M+1)")
ax3.legend()
ax3.set_title("Graph of Number of Valid Intervals")
 """
plt.tight_layout()
plt.show()


# fun: The optimal value of the objective function obtained after the optimization process
# x: The optimal solution, i.e., the values of the variables that minimize the objective function
# nit: The number of iterations performed during the optimization process
# jac: The Jacobian of the objective function at the optimal solution. This represents the gradient of the objective function with respect to each variable
# nfev: The number of evaluations of the objective function
# njev: The number of evaluations of the Jacobian (gradient) of the objective function