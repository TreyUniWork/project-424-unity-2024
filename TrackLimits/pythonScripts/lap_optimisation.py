from cornerModel_fit import produce_path
from findLimits import check_inside
from scipy.optimize import differential_evolution, dual_annealing
import numpy as np
from plotTrack import plot_autopilot
from acceleration_model import find_inputs

def laptime_optimiser(consts,header, sides, autopilot_data, auto_data, file_prefix, max_time):
    """_summary_

    Args:
        header (_type_): _description_
        sides (_type_): _description_
        autopilot_data (_type_): _description_
        auto_data (_type_): _description_
        file_prefix (_type_): _description_
        consts (_type_): _description_
        max_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    consts = consts.reshape((-1,2))

    c_inds, new_autopilot, new_auto = produce_path(autopilot_data, consts, auto_data, sides)

    if isinstance(c_inds, str):
        store_obj(400)
        return max_time*2

    ## matt's part
    new_inputs, obj = find_inputs(new_auto, c_inds)

    autopilot_data[:-120,17:19] = new_inputs[:-120]

    store_obj(obj)

    # print(f"Predicted Time: {pred_time}")
    # ## Write files necessary

    # np.savetxt(file_prefix+"autopilot_inprogress.csv", new_autopilot, delimiter=',', newline='\n', header=header, fmt="%.5f")

    # ## get outcome
    # print("Laptime recorded:\t\t\t\nType 0 for infeasible (crashes)")

    # ##delete all files after 19/09/2022
    # ##"C:\Users\lachl\OneDrive\Documents\Coding\project-424-unity\Assets\Replays"

    # obj = input()

    # if obj.isdigit():
    #     obj = float(obj)
    # else:
    #     return 2*max_time

    # if obj <300 or obj>400:
    #     return 2*max_time

    return obj

def store_obj(obj):

    with open("trackData\\testPaths\\functionObjective.txt", "a") as obj_file:
        
        obj_file.write(str(obj)+"\n")
    
        obj_file.close()

def callback(xk,f, convergence):

    with open("trackData\\testPaths\\optimisationResults.txt", "a") as sol_file:
        
        sol_file.write(str(xk)+"\n")
    
        sol_file.close()

    with open("trackData\\testPaths\\optimisationObjective.txt", "a") as obj_file:
        
        obj_file.write(str(f)+"\n")
    
        obj_file.close()


def main(track_name = "nordschleife", margin = 0.1):
    """_summary_

    Args:
        track_name (str, optional): _description_. Defaults to "nordschleife".
        margin (float, optional): _description_. Defaults to 0.1.
    """
    file_prefix = "C:\\Users\\lachl\\OneDrive\\Documents\\PERRINN 424\\Lap Data\\"
    file_name = "trackData\\testPaths\\initial_path.csv"

    consts = np.genfromtxt("trackData\\autopilot\\cornerModel_constants.csv", delimiter=",")

    with open(file_name,'r') as auto_file:
        header = auto_file.read().split('\n')
        header = '\n'.join(header[:2])
    
    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)
    auto_data = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',', dtype=float)

    sideL = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", delimiter=',', dtype=float)
    sideR = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", delimiter=',', dtype=float)
    sides = [sideL, sideR]

    x0 = consts.flatten()

    time = 308.327

    lower = x0 -margin
    upper = x0 +margin
    bounds = np.column_stack((lower, upper))

    solution = dual_annealing(laptime_optimiser, bounds=bounds, args=(header, sides, autopilot_data, auto_data, file_prefix, time), x0=x0,maxfun=3000, callback=callback, no_local_search=True)

    final_consts = np.reshape(solution.x,(-1,2))

    c_inds, new_autopilot_data, new_auto_data = produce_path(autopilot_data, final_consts, auto_data)
    new_inputs, pred_time = find_inputs(auto_data, c_inds)

    autopilot_data[:-120,17:19] = new_inputs[:-120]

    np.savetxt(file_prefix+"optimised_autopilot.csv", new_autopilot_data, delimiter=',', newline='\n', header=header, fmt="%.5f")

if __name__ == "__main__":

    main()