import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from matplotlib import pyplot as plt
import seaborn as sns

def corner_diff_wrapper(c, consts, corner_data, mode = "1"):

    return corner_diff(consts, corner_data, c, mode)

def corner_diff(consts, corner_data, c = None, mode = "0"):
    """_summary_

    Args:
        ca (_type_): _description_
        c1 (_type_): _description_
        corner_data (_type_): _description_
        return_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if mode != "1":
        c = minimize_scalar(fun=corner_diff_wrapper, method="brent", bracket=(-0.1,0.1), args = (consts, corner_data, "1")).x

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

    if mode == "0":

        mse = np.sum(np.sqrt((k-corner_data[3])**2)) / u.size
        return mse

    else:
            
        psi = calc_heading(u, k, ka, k0, k1, l0, ca, c1, psi0, corner_dist)
        x, y = calc_position(u, k, psi, x_data[0], y_data[0])

        if mode == "1":

            ## perpindicular distance

            delta = [x[-1]-x_data[-1], y[-1] - y_data[-1]]

            psi_end = psi[-1]
            y = np.tan(psi_end)

            vec = np.array([1, y])    # + +
            mag = np.sqrt(1+y**2)
            vec /= mag

            if psi_end > np.pi/2 and psi_end < np.pi*3/2:
                vec *= -1

            a = np.zeros((2,2))
            a[0] = vec
            a[1] = vec[::-1]

            parr, perp = np.linalg.solve(a, delta)

            return perp

        else:
            return np.array([corner_data[0], x, y, k, psi, corner_data[-1]]), u_indices, (parr>0)

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

def fit_corner_model(auto_data):
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
        consts[i] = differential_evolution(func=corner_diff, bounds=((-2,1),(-1,1)), args = (temp,"0", "0"), seed=88, x0=[0,0]).x

    np.savetxt("trackData\\autopilot\\cornerModel_constants.csv", consts, delimiter=",", newline="\n")

    return consts

def produce_path(autopilot_data, consts, auto_data):
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
        new_data, corner_inds, n_pa = corner_diff(cnt, temp_data, None, None)

        corner_inds += st_ind

        if n_pa:
            
            end_ind = np.where(corner_mask[corner_inds[-1]:])
            straight_data = auto_data[1:3, corner_inds[-1]+1:corner_inds[-1]+end_ind]
            temp = replace_straights(new_data[4], new_data[1:3,-1], straight_data)

            if temp == "0":
                return "0", "0", "0"

            auto_data[corner_inds[-1]+1:corner_inds[-1]+end_ind] = temp
            
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
    straight_data = check_overlap(psi, start_pos, straight_data)

    if straight_data == "0":

        max_angle *= np.pi/180
        angles = np.linspace(0,max_angle,n+1)[1:]

        angles = np.ravel(np.column_stack((angles, angles*-1)))

        for ang in angles:

            straight_data = check_overlap(psi+ang, start_pos, straight_data)

            if straight_data != "0":
                return straight_data

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
    
    y = np.tan(psi)

    vec = np.array([1, y])    # + +
    mag = np.sqrt(1+y**2)
    vec /= mag

    for d in np.linspace(0,m,n):

        pos0 = start_pos+vec*d

        dist = np.sum(np.square(straight_data - pos0), axis=1)

        pos1_ind, pos2_ind = np.argsort(dist)[:2]

        pos1 = straight_data[pos1_ind]; pos2 = straight_data[pos2_ind]

        dist = np.abs((pos2[0]-pos1[0])*(pos1[1]-pos0[1]) - (pos1[0]-pos0[0])*(pos2[1]-pos1[1])) / np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)

        if dist < tol:
            first_ind = min(pos1_ind, pos2_ind)
            straight_data[:min(pos1_ind, pos2_ind)] = np.array([start_pos + vec*d2 for d2 in np.linspace(0,d,first_ind+1)])

        else:
            return "0"

def steeringAngle_fit(y, x):

    y_diff = y - np.mean(y)
    x_diff = x - np.mean(x)

    b1 = np.sum(x_diff*y_diff)/np.sum(x_diff**2)

    b0 = np.mean(y) - b1 * np.mean(x)

    return b0, b1

def plot_corner(corner_data, original_data, num, track_name):
    """_summary_

    Args:
        corner_data (_type_): _description_
        original_data (_type_): _description_
        num (_type_): _description_
        track_name (_type_): _description_
    """
    sns.set_theme()

    fig, axes = plt.subplots(nrows=2, ncols=2)

    plt.suptitle(track_name + " Corner " + str(num))

    axes[0,0].plot(*original_data[1:3], "r--", label="Original")
    axes[0,0].plot(*corner_data[1:3], "b--", label= "Model Fit")

    axes[1,0].plot(original_data[0], original_data[1] - original_data[1,0], "r--", label = "Original x")
    axes[1,0].plot(original_data[0], original_data[2] - original_data[2,0], "r-", label = "Original y")
    axes[1,0].plot(corner_data[0], corner_data[1] - corner_data[1,0], "b--", label = "Model x")
    axes[1,0].plot(corner_data[0], corner_data[2] - corner_data[2,0], "b-", label = "Model y")

    axes[0,1].plot(*original_data[[0,3]], "r--", label = "Original")
    axes[0,1].plot(*corner_data[[0,3]], "b--", label = "Model")

    axes[1,1].plot(*original_data[[0,4]], "r--", label = "Original")
    axes[1,1].plot(*corner_data[[0,4]], "b--", label = "Model")

    axes[0,0].set_xlabel("x (m)")
    axes[0,0].set_ylabel("y (m)")

    axes[1,0].set_xlabel("u (-)")
    axes[1,0].set_ylabel("delta pos (m)")

    axes[0,1].set_xlabel("u (-)")
    axes[0,1].set_ylabel("k (1/m)")

    axes[1,1].set_xlabel("u (-)")
    axes[1,1].set_ylabel("heading (rad)")

    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()
    axes[1,1].legend()

    plt.tight_layout()
    plt.show()


def main(track_name):
    """_summary_

    Args:
        track_name (_type_): _description_
    """

    file_name = "trackData\\testPaths\\2022-07-11 04.07.43 UTC 05.08.327 ideal.csv"
    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)
    auto = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',', dtype=float).T

    #consts = fit_corner_model(auto)

    consts = np.genfromtxt("trackData\\autopilot\\cornerModel_constants.csv", delimiter=",")

    c_inds, autopilot_data, auto_data = produce_path(autopilot_data, track_name, consts, auto)

    with open(file_name,'r') as auto_file:
        header = auto_file.read().split('\n')
        header = '\n'.join(header[:2])

    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_cornerIndices.csv", np.array(c_inds, dtype=int), delimiter=',', newline='\n', fmt="%d")
    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_cornerModel_original.csv", autopilot_data, delimiter=',', newline='\n', header=header, fmt="%.5f")
    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_autodata.csv", auto_data, delimiter=',', newline='\n', fmt="%.5f")

if __name__ == "__main__":

    main("nordschleife")
