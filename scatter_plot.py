import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import matplotlib.pyplot as plt

def objective(y:np.ndarray, Z:int, p:list, M:int, N:int, avg_aois:dict={}) -> float:
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
            paralellogram_area += y[i] * y[j] * (p[j-1])**(j - i) # I am using p as a error probability

    objective = (triangle_area + Z*paralellogram_area)/N
    avg_aoi = (triangle_area + paralellogram_area)/N
    avg_aois[Z] = avg_aoi # save the average AoI with Z as key
    
    return objective

## ALL PARAMETERS ARE HERE
M = 3 # Number of updates
N = 1  # Time interval

open("update:"+str(M)+"different prob.txt", 'w').close()

additional_probs = [0, 0.02, 0.04]
main_p = 0.3 # when we use different probabilities for same update number, they start to converge earlier

## ALL PARAMETERS ARE HERE

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


for p1 in additional_probs:
    p_list = [[p1 if i == j else main_p for i in range(M)] for j in range(M)]

    colormap = plt.cm.get_cmap('tab10')
    # creating 3 separate graphs thanks to different axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    x_values = [k for k in range(1, M + 1)]

    line_styles = ['-', '--', '-.', ':']

    file1 = open("update:"+str(M)+"different prob.txt",'a')

    for i, z in enumerate(Z_list):
        result_values = []  # Store the result values for each p in p_list
        avg_aoi_values = []  # Store the avg_aoi values for each p in p_list
        interval_values = []

        best_perm_idx_aoi = -1  # Index of the best permutation for average age of info
        best_perm_idx_penalty = -1 # Index of the best permutation for penalty
        best_avg_aoi = float('inf')  # Initialize with a large value
        best_avg_pen = float('inf')


        for j, p in enumerate(p_list):
            print(p_list)
            print(p)
            result = minimize(lambda ys: objective(ys, z, p, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
            result_values.append(result.fun)  # Append the result value
            avg_aoi = AoIs[z]  # Calculate the avg_aoi value
            avg_aoi_values.append(avg_aoi)
            
            valid_interval = sum([1 for val in result.x if val > 1e-5])
            interval_values.append(valid_interval)
            num_of_updates = valid_interval-1

            file1.write(f"Z:{z} ")
            file1.write(f"p:{p} ")
            file1.write(f"Number of Updates:{num_of_updates} ")
            file1.write(f"Average Age of Information with penalty:{result.fun:.5f} ")
            file1.write(f"Average Age of Information without penalty:{AoIs[z]:.5f} ")
            file1.write(f"Optimal interupdate times: {result.x} \n") 

            if avg_aoi < best_avg_aoi:
                best_avg_aoi = avg_aoi
                best_perm_idx_aoi = j
            
            if result.fun < best_avg_pen:
                best_avg_pen = result.fun
                best_perm_idx_penalty = j

        file1.write("\n")

        cutting_idx = 0
        my_bool = False

        #print(interval_values)
        for j in range(len(interval_values)):
            if interval_values[j] <= M:
                my_bool = True
                cutting_idx = j
                break
        #print(my_bool)
        #print(cutting_idx)

        color = colormap(i)

        if my_bool:
            ax1.scatter(x_values[:cutting_idx], result_values[:cutting_idx], label="Z = " + str(z), marker='o', facecolors='none', edgecolors=color)
            ax1.scatter(x_values[best_perm_idx_penalty], result_values[best_perm_idx_penalty], label="Best permutation (Z = " + str(z) + ")", marker='D', s=50, facecolors='none', edgecolors=color)
            ax2.scatter(x_values[:cutting_idx], avg_aoi_values[:cutting_idx], label="Z = " + str(z), marker='o', facecolors='none', edgecolors=color)
            ax2.scatter(x_values[best_perm_idx_aoi], avg_aoi_values[best_perm_idx_aoi], label="Best permutation (Z = " + str(z) + ")", marker='D', s=50, facecolors='none', edgecolors=color)
        else:
            ax1.scatter(x_values, result_values, label="Z = " + str(z), marker='o', facecolors='none', edgecolors=color)
            ax1.scatter(x_values[best_perm_idx_penalty], result_values[best_perm_idx_penalty], label="Best permutation (Z = " + str(z) + ")", marker='D', s=50, facecolors='none', edgecolors=color)
            ax2.scatter(x_values, avg_aoi_values, label="Z = " + str(z), marker='o', facecolors='none', edgecolors=color)
            ax2.scatter(x_values[best_perm_idx_aoi], avg_aoi_values[best_perm_idx_aoi], label="Best permutation (Z = " + str(z) + ")", marker='D', s=50, facecolors='none', edgecolors=color)
        
        ax1.grid(True)
        ax2.grid(True)
        
        ax3.plot(x_values, interval_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
        # if we want to plot whole graphs just comment the last part from print and use last 3 line

    x_ticks = [k for k in range(1, M + 1)]
    x_tick_labels = [str(x)+". permutation" for x in x_ticks]

    # Customize the tick labels on the x-axis
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_tick_labels)

    ax1.set_xlabel("Probability permutations")
    ax1.set_ylabel("Minimal Penalty")
    ax1.set_title(f"Min. Penalty with update number of: {M} with p:{main_p}, p1:{p1}")

    ax2.set_xlabel("Probability permutations")
    ax2.set_ylabel("Avg AOI")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax2.set_title(f"Avg AOI with update number of: {M} with p:{main_p}, p1:{p1}")

    ax3.set_xlabel("Probability permutations")
    ax3.set_ylabel("Num. of Valid Intervals\n(M+1)")
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax3.set_title("Graph of Number of Valid Intervals")

    plt.tight_layout()
    plt.show()
