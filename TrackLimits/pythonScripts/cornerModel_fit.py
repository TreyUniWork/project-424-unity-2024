from os import times
import numpy as np
from scipy.optimize import minimize_scalar, dual_annealing
from plotTrack import plot_autopilot, plot_corner, plot_cornerSegments, plot_steeringFit, plot_cornerLength, plot_optimiser, plot_solution,plot_constBounds
from findLimits import check_inside
from acceleration_model import find_inputs

def corner_diff_wrapper(c, consts, corner_data, sides):

    return corner_diff(consts, corner_data, c, "1", sides)

def corner_diff(consts, corner_data, c, mode, sides):
    """_summary_

    Args:
        ca (_type_): _description_
        c1 (_type_): _description_
        corner_data (_type_): _description_
        return_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if not isinstance(c,float):
        c = minimize_scalar(fun=corner_diff_wrapper, method="bounded", bounds=(-0.1,0.1), args = (consts, corner_data, sides)).x

    k0 = corner_data[3,0]
    k1 = corner_data[3,-1]

    ca = 10**(consts[0])
    c1 = 10**(consts[1])

    x_data = corner_data[1]
    y_data = corner_data[2]

    u0 = corner_data[0,0]
    u1 = corner_data[0,-1]

    u_range = u1 - u0

    u = (corner_data[0] - u0) / u_range

    psi0 = corner_data[4,0]
    psi_1 = corner_data[4,-1]

    del_psi = (psi_1 - psi0)

    psi_diffs = np.ediff1d(corner_data[4])

    if np.sum((np.abs(psi_diffs)>1e0))>0:
        del_psi += 2*np.pi * np.sign(np.mean(corner_data[3]))
        psi_1 = psi0+del_psi

    l0 = 1/(1+c1+ca)

    u_segments = np.array([0, l0, l0 * (1+ca), 1]) # replace closest point in u

    u_indices = np.searchsorted(u, u_segments, side="left")

    #pick corner distance to minimise end point separation
    corner_dist = u_range * (74559.27 / 3.6) * 10 ** c # c = 0 (std)

    ka = ((2*del_psi/(l0 * corner_dist)) - (k0 + c1*k1)) / (1 + c1 + 2*ca)
    k = calc_curvature(u, ka, k0, k1, l0, c1)

    psi = calc_heading(u, k, ka, k0, k1, l0, ca, c1, psi0, corner_dist)
    x, y = calc_position(u, k, psi, x_data[0], y_data[0])

    if mode == "0":
        mse = np.sum(np.sqrt((k-corner_data[3])**2)) / u.size
        isInside = check_inside(np.array([x,y]),sides)
        return mse*(1+(1-isInside))

    else:

        ## perpindicular distance
        delta = [x[-1]-x_data[-1], y[-1] - y_data[-1]]

        psi_end = psi[-1] % (2*np.pi)
        y_dir = np.tan(psi_end)

        vec = np.array([1, y_dir])    # + +
        mag = np.sqrt(1+y_dir**2)
        vec /= mag

        if psi_end > np.pi/2 and psi_end < np.pi*3/2:
            vec *= -1

        a = np.zeros((2,2))
        a[0] = vec
        a[1] = vec[::-1]
        a[1,1] *=-1

        parr, perp = np.linalg.solve(a, delta)

        if mode == "1":
            return np.abs(perp)
        else:
            isInside = check_inside(np.array([x,y]),sides)
            if isInside:
                return np.array([corner_data[0], x, y, k, psi, corner_data[-1]]), u_indices, (parr>0)
            else:
                return "0", "0", "0"
            

def calc_steering_angle(beta, k):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """

    theta = beta[0] + beta[1] * k

    return theta

def calc_curvature(u, ka, k0, k1, l0, c1):
    """_summary_

    Args:
        u (_type_): _description_
        ka (_type_): _description_
        k0 (_type_): _description_
        k1 (_type_): _description_
        l0 (_type_): _description_
        c1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    k = np.zeros(u.shape)

    clothoid0_mask = (u<l0)
    arc_mask = (u<1-l0*c1) & (u>=l0)
    clothoid1_mask = (u>=(1-l0*c1))

    k[clothoid0_mask] = k0 + ((ka-k0)/l0) * u[clothoid0_mask]
    k[arc_mask] = ka
    k[clothoid1_mask] = ((k1*l0*c1 + ka - k1) + (k1-ka) * u[clothoid1_mask]) / (l0*c1)

    return k

def calc_heading(u, k, ka, k0, k1, l0, ca, c1, psi0, corner_dist):
    """_summary_

    Args:
        u (_type_): _description_
        k (_type_): _description_
        ka (_type_): _description_
        k0 (_type_): _description_
        k1 (_type_): _description_
        l0 (_type_): _description_
        ca (_type_): _description_
        c1 (_type_): _description_
        psi0 (_type_): _description_
        psi_1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    psi = np.zeros(u.shape)

    clothoid0_mask = (u<l0)
    arc_mask = (u<1-l0*c1) & (u>=l0)
    clothoid1_mask = (u>=(1-l0*c1))

    psi1 = psi0 + (ka+k0) * l0 * corner_dist / 2
    psi2 = psi1 + ka * l0 * corner_dist * ca
    psi3 = psi2 + (ka+k1) * l0* corner_dist * c1 / 2

    psi[clothoid0_mask] = psi0 + (k[clothoid0_mask]+k0) * u[clothoid0_mask] * corner_dist/2

    #TODO
    psi[arc_mask] = psi1 + ka * (u[arc_mask] - l0) * corner_dist

    #TODO 
    psi[clothoid1_mask] = psi2 + (k[clothoid1_mask] + ka) * (u[clothoid1_mask] - (1-l0*c1)) * corner_dist/2

    return psi

def calc_position(u, k, psi, x0, y0):
    """_summary_

    Args:
        u (_type_): _description_
        k (_type_): _description_
        psi (_type_): _description_
        x0 (_type_): _description_
        y0 (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = np.zeros(u.shape)
    y = np.zeros(u.shape)

    x[0] = x0
    y[0] = y0

    x[1:] = 1/k[1:] * (np.sin(psi[1:]) - np.sin(psi[:-1]))
    y[1:] = 1/k[1:] * (np.cos(psi[:-1]) - np.cos(psi[1:]))

    x = x.cumsum()
    y = y.cumsum()

    return x, y

def fit_corner_model(auto_data,sides):
    """_summary_

    Args:
        auto_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = int(np.max(auto_data[-1])) +1

    consts = np.zeros((n,2))

    for i in range(n):
        
        temp = auto_data[:,(auto_data[-1] == i)]
        consts[i] = dual_annealing(func=corner_diff, bounds=((-2,2),(-2,2)), args = (temp,"0", "0",sides), seed=88, x0=[0,0], no_local_search=True, maxfun=10000, maxiter=1000).x

    np.savetxt("trackData\\autopilot\\cornerModel_constants.csv", consts, delimiter=",", newline="\n")

    return consts

def produce_path(autopilot_data, consts, auto_data, sides):
    """_summary_

    Args:
        file_name (_type_): _description_
        track_name (_type_): _description_
        consts (_type_): _description_
        auto_data (_type_): _description_
    """
    c_inds = []

    beta = steeringAngle_fit(autopilot_data[:,16], auto_data[3])

    for i, cnt in enumerate(consts):

        corner_mask = (auto_data[-1] == i)
        st_ind = np.argwhere(corner_mask)[0,0]
        temp_data = auto_data[:,corner_mask]
        new_data, corner_inds, n_pa = corner_diff(cnt, temp_data, None, None, sides)

        if isinstance(n_pa,str):
            return "0", "0", "0"
            #continue

        corner_inds += st_ind
        # if i in [0,22]:
        #     pass
        #plot_corner(new_data, temp_data, f"Nordschleife Corner Fit {i}")

        if n_pa:
            
            isEnd = False
            if i == consts.shape[0]-1:
                straight_data = auto_data[1:3, corner_inds[-1]+1:]
                isEnd=True

            else:

                next_corner_mask = (auto_data[-1] == i+1)
                end_ind = np.where(next_corner_mask[corner_inds[-1]:])[0]
                st = corner_inds[-1]+1
                nd = corner_inds[-1]+end_ind[0]
                straight_data = auto_data[1:3, st:nd]
            straight_data = replace_straights(new_data[4,-1], new_data[1:3,-1], straight_data)

            if isinstance(straight_data, str):
                return "0", "0", "0"

            if isEnd:
                auto_data[1:3,corner_inds[-1]+1:] = straight_data
                isEnd = False
            else:   
                auto_data[1:3,st:nd] = straight_data
            
        auto_data[:,corner_mask] = new_data
        c_inds.append(corner_inds)

    autopilot_data[:,10] = auto_data[1]
    autopilot_data[:,12] = auto_data[2]

    autopilot_data[:,16] = calc_steering_angle(beta, auto_data[3])

    return np.array(c_inds), autopilot_data, auto_data

def replace_straights(psi, start_pos, straight_data, max_angle = 10, n = 10):
    """_summary_

    Args:
        psi (_type_): _description_
        start_pos (_type_): _description_
        straight_data (_type_): _description_
        max_angle (int, optional): _description_. Defaults to 10.
        n (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    straight_data_temp = check_overlap(psi, start_pos, straight_data)

    if isinstance(straight_data_temp,str):

        max_angle *= np.pi/180
        angles = np.linspace(0,max_angle,n+1)[1:]

        angles = np.ravel(np.column_stack((angles, angles*-1)))

        for ang in angles:

            straight_data_temp = check_overlap(psi+ang, start_pos, straight_data)

            if not isinstance(straight_data_temp,str):
                return straight_data_temp

        return "0"
    else:
        return straight_data

def check_overlap(psi, start_pos, straight_data, m = 100, n = 10000, tol = 1e-1):
    """_summary_

    Args:
        psi (_type_): _description_
        start_pos (_type_): _description_
        straight_data (_type_): _description_
        m (int, optional): _description_. Defaults to 100.
        n (int, optional): _description_. Defaults to 10000.
        tol (_type_, optional): _description_. Defaults to 1e-1.

    Returns:
        _type_: _description_
    """

    # have to refactor this shit, it is terrible
    
    y = np.tan(psi)

    vec = np.array([1, y])    # + +
    mag = np.sqrt(1+y**2)
    vec /= mag

    if straight_data.shape[1] == 1:
        return straight_data

    straight_data = straight_data.T

    options = np.linspace(0.01,m,n)[0:]

    for d in options:

        pos0 = start_pos+vec*d

        dist = np.sum(np.square(straight_data - pos0), axis=1)

        pos1_ind, pos2_ind = np.argsort(dist)[:2]

        pos1 = straight_data[pos1_ind]; pos2 = straight_data[pos2_ind]

        dist = np.abs((pos2[0]-pos1[0])*(pos1[1]-pos0[1]) - (pos1[0]-pos0[0])*(pos2[1]-pos1[1])) / np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)

        if dist < tol:
            first_ind = max(pos1_ind, pos2_ind)
            straight_data[:first_ind] = np.array([start_pos + vec*d2 for d2 in np.linspace(0,d,first_ind)])

            return straight_data.T

    return "0"

def steeringAngle_fit(y, x):

    y_diff = y - np.mean(y)
    x_diff = x - np.mean(x)

    b1 = np.sum(x_diff*y_diff)/np.sum(x_diff**2)

    b0 = np.mean(y) - b1 * np.mean(x)

    # plot_steeringFit(x,y, [b0,b1])

    return b0, b1

def get_solution_consts():

    with open("trackData\\testPaths\\optimisationResults.txt", "r") as sol_file:

        sol_consts = sol_file.read()
        sol_file.close()

    sol_consts = sol_consts.replace("\n", "").split("][")
    sol_consts[0] = sol_consts[0][1:];sol_consts[-1] = sol_consts[-1][:-1]

    sol_consts = [[float(x) for x in ln.split()] for ln in sol_consts]

    sol_consts = [np.reshape(ln, (-1,2)) for ln in sol_consts]

    return np.array(sol_consts)[-1]

def main(track_name):
    """_summary_

    Args:
        track_name (_type_): _description_
    """

    file_name = "trackData\\testPaths\\initial_path.csv"
    old_autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)
    auto = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',', dtype=float)

    sideL = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", delimiter=',', dtype=float)
    sideR = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", delimiter=',', dtype=float)
    sides = [sideL, sideR]

    consts = fit_corner_model(auto, sides)

    consts = np.genfromtxt("trackData\\autopilot\\cornerModel_constants.csv", delimiter=",")

    const_best = get_solution_consts()

    c_inds, autopilot_data, auto_data = produce_path(old_autopilot_data, const_best, auto.copy(), sides)
    autopilot_data[:,17:19], pred_time = find_inputs(auto_data, c_inds)

    with open(file_name,'r') as auto_file:
        header = auto_file.read().split('\n')
        header = '\n'.join(header[:2])

    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_cornerIndices.csv", np.array(c_inds, dtype=int), delimiter=',', newline='\n', fmt="%d")
    np.savetxt("C:\\Users\\lachl\\OneDrive\\Documents\\PERRINN 424\\Lap Data\\"+track_name + "_test.csv", autopilot_data, delimiter=',', newline='\n', header=header, fmt="%.5f")
    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_autodata.csv", auto_data, delimiter=',', newline='\n', fmt="%.5f")

if __name__ == "__main__":

    main("nordschleife")
