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

M = 3 # Number of updates
N = 1  # Time interval

p1_list = [0, 0.03, 0.05, 0.07, 0.1]
for p1 in p1_list:
    main_p =  np.linspace(p1,0.5,20) # For M=3, Z=2 around p=0.4 we can see that number of updates drop

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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    x_values = np.linspace(p1, 0.5, len(main_p))

    line_styles = ['-', '--', '-.', ':']

    file1 = open("update:"+str(M)+"different prob.txt",'w')

    for i, z in enumerate(Z_list):
        result_values = []  # Store the result values for each p in p_list
        avg_aoi_values = []  # Store the avg_aoi values for each p in p_list
        how_many_updates = []

        for j, p in enumerate(main_p):
            p_list = [[p1 if a == b else p for a in range(M)] for b in range(M)]

            best_result_value = float('inf')
            best_avg_aoi_value = float('inf')
            less_than_M = 0
            complete_update = 0
            my_bool = False
            
            for lst in p_list:
                result = minimize(lambda ys: objective(ys, z, lst, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
                
                # Here, i am looking a permutation used how many updates, if it is less than M i increment the variable
                # If all the permutations used less than M, then we should cut the plots here
                num_of_updates = sum([1 for val in result.x if val > 1e-5])-1
                if num_of_updates < M:
                    less_than_M += 1
                if less_than_M == M:
                    cutting_idx = j
                    my_bool = True
                    break

                # If a permutation uses all updates, even if other versions doesn't use we should uuse it's value as a best result
                if num_of_updates == M:
                    if result.fun < best_result_value:
                        best_result_value = result.fun
                    if AoIs[z] < best_avg_aoi_value:
                        best_avg_aoi_value = AoIs[z]
                    complete_update += 1

                file1.write(f"Z:{z} ")
                file1.write(f"p:{lst} ")
                file1.write(f"Number of Updates:{num_of_updates} ")
                file1.write(f"Average Age of Information with penalty:{result.fun:.5f} ")
                file1.write(f"Average Age of Information without penalty:{AoIs[z]:.5f} ")
                file1.write(f"Optimal interupdate times: {result.x} \n") 


            result_values.append(best_result_value)  # Append the result value
            avg_aoi_values.append(best_avg_aoi_value)  # Calculate and append the avg_aoi value
            how_many_updates.append(complete_update) # How many permutation used all updates

        file1.write("\n")
        
        if my_bool:
            ax1.plot(x_values[:cutting_idx], result_values[:cutting_idx], label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
            ax2.plot(x_values[:cutting_idx], avg_aoi_values[:cutting_idx], label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
        else:
            ax1.plot(x_values, result_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
            ax2.plot(x_values, avg_aoi_values, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
        ax3.plot(x_values, how_many_updates, label="Z = " + str(z), linestyle=line_styles[i % len(line_styles)], color='blue')
        # if we want to plot whole graphs just comment the last part from print and use last 3 line

    x_ticks = np.linspace(p1, 0.5, 6) # We should start the leftmost point from p1
    decimal_places = 2
    x_ticks = np.around(x_ticks, decimals=decimal_places)
    x_tick_labels = [str(x) for x in x_ticks]

    # Customize the tick labels on the x-axis
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)

    ax1.set_xlabel("Probability (p)")
    ax1.set_ylabel("Minimal Penalty Value")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax1.set_title(f"Graph of the Minimal Penalty with update number of: {M} with p1: {p1}")

    ax2.set_xlabel("Probability (p)")
    ax2.set_ylabel("Avg AOI")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax2.set_title(f"Graph of Avg AOI with update number of: {M} with p1: {p1}")

    ax3.set_xlabel("Probability (p)")
    ax3.set_ylabel(f"Number of updates")
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax3.set_title(f"Number of Permutations that makes {M} update")

    plt.tight_layout()
    plt.show()


# fun: The optimal value of the objective function obtained after the optimization process
# x: The optimal solution, i.e., the values of the variables that minimize the objective function
# nit: The number of iterations performed during the optimization process
# jac: The Jacobian of the objective function at the optimal solution. This represents the gradient of the objective function with respect to each variable
# nfev: The number of evaluations of the objective function
# njev: The number of evaluations of the Jacobian (gradient) of the objective function