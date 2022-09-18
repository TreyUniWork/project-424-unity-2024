from cornerModel_fit import corner_diff, produce_path
from scipy.optimize import differential_evolution
import numpy as np



def laptime_optimiser(consts, autopilot_data, auto_data, sides, max_time):

    consts = consts.reshape((-1,2))

    c_inds, new_autopilot, new_auto = produce_path(autopilot_data, consts, auto_data)

    if not check_inside(new_auto, sides) or isinstance(c_inds, str):

        return max_time*1.1


    ## matt's part




    return obj



def main(margin = 0.1)

    autopilot_data = 
    auto_data = 
    sides = 
    consts = 

    x0 = consts.flatten()

    lower = consts * (1-margin)
    upper = consts * (1+margin)
    bounds = np.column_stack(lower, upper)

    solution = differential_evolution(laptime_optimiser, bounds=bounds, args=(autopilot_data, auto_data, sides), x0=x0, maxiter=100)

    



if __name__ == "__main__":

    main()