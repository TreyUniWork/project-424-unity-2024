from cornerModel_fit import produce_path
from scipy.optimize import differential_evolution
import numpy as np

def laptime_optimiser(header, sides, autopilot_data, auto_data, file_prefix, consts, max_time):

    consts = consts.reshape((-1,2))

    c_inds, new_autopilot, new_auto = produce_path(autopilot_data, consts, auto_data)

    if not check_inside(new_auto, sides) or isinstance(c_inds, str):

        return max_time*1.1


    ## matt's part



    ## Write files necessary

    np.savetxt(file_prefix+"autopilot_inprogress.csv", new_autopilot, delimiter=',', newline='\n', header=header, fmt="%.5f")

    print("Laptime recorded:\t\t\t\nType 0 for infeasible (crashes)")

    obj = input()

    if obj == "0" or not isinstance(float(obj, float)):
        return max_time*1.1

    return float(obj)

def check_inside(auto_data, sides):

    pass


def main(track_name = "nordschleife", margin = 0.1):

    file_prefix = "C:\\Users\\lachl\\OneDrive\\Documents\\PERRINN 424\\Lap Data\\"
    file_name = "TrackLimits\\trackData\\testPaths\\initial_path.csv"

    consts = np.genfromtxt("trackData\\autopilot\\cornerModel_constants.csv", delimiter=",")

    with open(file_name,'r') as auto_file:
        header = auto_file.read().split('\n')
        header = '\n'.join(header[:2])
    
    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)
    auto_data = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',', dtype=float).T

    sideL = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", delimiter=',', dtype=float)
    sideR = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", delimiter=',', dtype=float)
    sides = [sideL, sideR] 

    x0 = consts.flatten()

    time = 308.327

    lower = x0 * (1-margin)
    upper = x0 * (1+margin)
    bounds = np.column_stack(lower, upper)

    solution = differential_evolution(laptime_optimiser, bounds=bounds, args=(header, sides,autopilot_data, auto_data, file_prefix, consts, time), x0=x0, maxiter=100)

    final_consts = solution.x.reshape((-1,2))

    c_inds, new_autopilot_data, new_auto_data = produce_path(autopilot_data, final_consts, auto_data)

    np.savetxt(file_prefix+"optimised_autopilot.csv", new_autopilot_data, delimiter=',', newline='\n', header=header, fmt="%.5f")

if __name__ == "__main__":

    main()