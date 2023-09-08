import matplotlib
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
""" marker_translations = {-1: '', # no result -> no marker
                       0: 'o',
                       1: 's',
                       2: 'D'} """
color_translations = {-1: 'b', # no result -> no color
                      0: 'k',
                      1: 'g',
                      2: 'r'}

p1_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1]

for p1 in p1_list:
    file1 = open("outputs/update/update:"+str(M)+"prob:"+str(p1)+"different prob.txt",'w')
    main_p =  np.linspace(p1,0.5,100) # For M=3, Z=2 around p=0.4 we can see that number of updates drop
    # did a finer scale for Z=10
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

    x_values = np.linspace(p1, 0.5, len(main_p))

    #line_styles = ['-', '--', '-.', ':']
    line_colors = ['blue','orange','purple','brown']
    marker_styles = ['o', 'p', '^', 's'] 

    for i, z in enumerate(Z_list):

        result_values = []  # Store the result values for each p in p_list
        avg_aoi_values = []  # Store the avg_aoi values for each p in p_list
        how_many_updates = []
        markers = [] # Store the markers of the best permutation
        my_idx = 0
        cut_idx = 0
        cut_bool = True

        for p in main_p:
            # burda tüm permutasyonların kombinasyonları bulunuyor
            p_list = [[p1 if a == b else p for a in range(M)] for b in range(M)]

            best_result_value = np.inf
            best_avg_aoi_value = np.inf
            dashed_result = 0
            dashed_aoi = 0
            best_marker = -1
            less_than_M = 0
            complete_update = 0
            
            for idx, lst in enumerate(p_list):
                # lst permutasyonlardan biri
                result = minimize(lambda ys: objective(ys, z, lst, M, N, AoIs), y0, bounds=bounds, constraints=linear_constraint)
                
                # Here, i am looking a permutation used how many updates, if it is less than M i increment the variable
                num_of_updates = sum([1 for val in result.x if val > 1e-5])-1
                """ if z==10 and p1==0.05:
                    print(my_idx)
                    print(num_of_updates) """
                if num_of_updates < M:
                    less_than_M += 1
                # all permutations used less than M, so we should cut the plots here
                if less_than_M == M:
                    if cut_bool:
                        cut_idx = my_idx
                        cut_bool = False
                    best_result_value = result.fun
                    best_avg_aoi_value = AoIs[z]

                # If a permutation uses all updates, even if other versions doesn't use we should use it's value as a best result
                if num_of_updates == M:
                    if result.fun < best_result_value:
                        best_result_value = result.fun
                        best_marker = idx # save the marker of the best permutation
                    if AoIs[z] < best_avg_aoi_value:
                        best_avg_aoi_value = AoIs[z]
                    complete_update += 1

                file1.write(f"Z:{z} ")
                file1.write(f"p:{lst} ")
                #file1.write(f"Number of Updates:{num_of_updates} ")
                file1.write(f"Average Age of Information with penalty:{result.fun:.5f} ")
                file1.write(f"Average Age of Information without penalty:{AoIs[z]:.5f} ")
                file1.write(f"Optimal interupdate times: {result.x} \n") 

            result_values.append(best_result_value)  # Append the result value
            avg_aoi_values.append(best_avg_aoi_value)  # Calculate and append the avg_aoi value
            how_many_updates.append(complete_update) # How many permutation used all updates
            my_idx += 1
            #markers.append(best_marker) # Append the marker of the best permutation

        file1.write("\n")
        # translate the markers
        #colors = [color_translations[m] for m in markers]
        #markers = [marker_translations[m] for m in markers]
        if cut_bool:
            ax1.plot(x_values, result_values, label="Z = " + str(z), color=line_colors[i % len(line_colors)], marker=marker_styles[i % len(marker_styles)], markevery=10)
            ax2.plot(x_values, avg_aoi_values, label="Z = " + str(z), color=line_colors[i % len(line_colors)], marker=marker_styles[i % len(marker_styles)], markevery=10)
        else:
            ax1.plot(x_values[:cut_idx+1], result_values[:cut_idx+1], label="Z = " + str(z), color=line_colors[i % len(line_colors)], marker=marker_styles[i % len(marker_styles)], markevery=10)
            ax1.plot(x_values[cut_idx:], result_values[cut_idx:], linestyle='--', label="Z = " + str(z), color=line_colors[i % len(line_colors)], marker=marker_styles[i % len(marker_styles)], markevery=10)
            # plot markers
            """ for x, y, m, c in zip(x_values, result_values, markers, colors):
                ax1.plot(x, y, marker=m, fillstyle='none', color=c) """

            ax2.plot(x_values[:cut_idx+1], avg_aoi_values[:cut_idx+1], label="Z = " + str(z), color=line_colors[i % len(line_colors)], marker=marker_styles[i % len(marker_styles)], markevery=10)
            ax2.plot(x_values[cut_idx:], avg_aoi_values[cut_idx:], linestyle='--', label="Z = " + str(z), color=line_colors[i % len(line_colors)], marker=marker_styles[i % len(marker_styles)], markevery=10)

    # Remember to close the file when you're done writing to it
    file1.close()

    x_ticks = np.linspace(0, 0.5, 6) # We should start the leftmost point from p1
    decimal_places = 2
    x_ticks = np.around(x_ticks, decimals=decimal_places)
    x_tick_labels = [str(x) for x in x_ticks]

    # Customize the tick labels on the x-axis
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax1.set_xlim(0, 0.5)
    ax2.set_xlim(0, 0.5)
    
    # fix the legend of ax1 with the markers
    """ handles, labels = ax1.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color='k', marker='o', linestyle='None', fillstyle='none'))
    handles.append(plt.Line2D([], [], color='g', marker='s', linestyle='None', fillstyle='none'))
    handles.append(plt.Line2D([], [], color='r', marker='D', linestyle='None', fillstyle='none'))
    labels.extend(["Permutation A", "Permutation B", "Permutation C"]) """

    ax1.set_ylabel("Minimal Penalty Value")
    ax1.legend(ncols=2, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax1.set_title(f"p1: {p1}")

    ax2.set_ylabel("Avg AOI")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=True)
    fig1.savefig(f"outputs/plots/penalty_p1_{p1}_updates_{M}.png", bbox_inches='tight', pad_inches=0)
    fig2.savefig(f"outputs/plots/aoi_p1_{p1}_updates_{M}.png", bbox_inches='tight', pad_inches=0)
    matplotlib.pyplot.close()

# fun: The optimal value of the objective function obtained after the optimization process
# x: The optimal solution, i.e., the values of the variables that minimize the objective function
# nit: The number of iterations performed during the optimization process
# jac: The Jacobian of the objective function at the optimal solution. This represents the gradient of the objective function with respect to each variable
# nfev: The number of evaluations of the objective function
# njev: The number of evaluations of the Jacobian (gradient) of the objective function