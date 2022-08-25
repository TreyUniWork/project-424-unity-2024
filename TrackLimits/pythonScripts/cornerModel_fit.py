import numpy as np
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
import seaborn as sns

def corner_diff(consts, corner_data, returnMse = False):
    """_summary_

    Args:
        ca (_type_): _description_
        c1 (_type_): _description_
        corner_data (_type_): _description_
        return_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    k0 = corner_data[3,0]
    k1 = corner_data[3,-1]

    ca = 10**consts[0]
    c1 = 10**consts[1]

    x_data = corner_data[1]
    y_data = corner_data[2]

    u0 = corner_data[0,0]
    u1 = corner_data[0,-1]

    u_range = u1 - u0

    u = (corner_data[0] - u0) / u_range

    corner_dist = u_range * 74559.27 / 3.5

    psi0 = corner_data[4,0]
    psi_1 = corner_data[4,-1]

    del_psi = (psi_1 - psi0)

    psi_diffs = np.ediff1d(corner_data[4])

    if np.sum((np.abs(psi_diffs)>1e0))>0:
        del_psi += 2*np.pi * np.sign(np.mean(corner_data[3]))
        psi_1 = psi0+del_psi

    l0 = 1/(1+c1+ca)

    ka = ((2*del_psi/(l0 * corner_dist)) - (k0 + c1*k1)) / (1 + c1 + 2*ca)

    k = calc_curvature(u, ka, k0, k1, l0, c1)

    if returnMse:

        mse = np.sum(np.sqrt((k-corner_data[3])**2)) / u.size
        return mse

    else:

        psi, psi_consts = calc_heading(u, k, ka, k0, k1, l0, ca, c1, psi0, psi_1, corner_dist)

        x, y = calc_position(u, k, psi, ka, k0, k1, l0, ca, c1, x_data[0], y_data[0], x_data[-1], y_data[-1], corner_dist, psi0, *psi_consts)

        return np.array([corner_data[0], x, y, k, psi, corner_data[-1]])

def calc_steering_angle(auto_data):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    parr_vector = np.array([np.tan(auto_data[4]),np.zeros(auto_data.shape[1])+1])

    old_coord = auto_data[1:3]

    b_vector = old_coord[:,1:] - old_coord[:,:-1]

    steering_angle = np.zeros(auto_data.shape[1])

    for i, (p_vec, b_temp) in enumerate(zip(parr_vector.T,b_vector.T)):

        a = np.array([p_vec, p_vec[::-1]])
        perp, parr = np.linalg.solve(a, b_temp)

        steering_angle[i] = np.arctan(perp/parr) * (180/np.pi)

    return steering_angle

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

def calc_heading(u, k, ka, k0, k1, l0, ca, c1, psi0, psi_1, corner_dist):
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

    assert abs(psi3-psi_1) < 1e-1

    psi[clothoid0_mask] = psi0 + (k[clothoid0_mask]+k0) * u[clothoid0_mask] * corner_dist/2

    #TODO
    psi[arc_mask] = psi1 + ka * (u[arc_mask] - l0) * corner_dist

    #TODO 
    psi[clothoid1_mask] = psi2 + (k[clothoid1_mask] + ka) * (u[clothoid1_mask] - (1-l0*c1)) * corner_dist/2

    return psi, (psi1, psi2, psi3)

def calc_position(u, k, psi, ka, k0, k1, l0, ca, c1, x0, y0, x_1, y_1, corner_dist, psi0, psi1, psi2, psi3):
    """_summary_

    Args:
        u (_type_): _description_
        k (_type_): _description_
        psi (_type_): _description_
        ka (_type_): _description_
        k0 (_type_): _description_
        k1 (_type_): _description_
        l0 (_type_): _description_
        ca (_type_): _description_
        c1 (_type_): _description_
        x0 (_type_): _description_
        y0 (_type_): _description_
        x_1 (_type_): _description_
        y_1 (_type_): _description_
        psi0 (_type_): _description_
        psi1 (_type_): _description_
        psi2 (_type_): _description_
        psi3 (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = np.zeros(u.shape)
    y = np.zeros(u.shape)

    x[0] = x0
    y[0] = y0

    for i in range(1, u.shape[0]):

        x[i] = x[i-1] + 1/k[i] * (np.sin(psi[i]) - np.sin(psi[i-1]))
        y[i] = y[i-1] + 1/k[i] * (np.cos(psi[i-1]) - np.cos(psi[i]))

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
        consts[i] = differential_evolution(func=corner_diff, bounds=((-2,1),(-1,1)), args = (temp,True), seed=88, x0=[0,0]).x

    np.savetxt("trackData\\autopilot\\cornerModel_constants.csv", consts, delimiter=",", newline="\n")

    return consts

def produce_path(file_name, track_name, consts, auto_data, plot = False):
    """_summary_

    Args:
        file_name (_type_): _description_
        track_name (_type_): _description_
        consts (_type_): _description_
        auto_data (_type_): _description_
    """
    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)

    for i, cnt in enumerate(consts):

        corner_mask = (auto_data[-1] == i)
        temp_data = auto_data[:,corner_mask]
        new_data = corner_diff(cnt, temp_data,False)

        if plot:
            plot_corner(new_data, temp_data, i, track_name)

        auto_data[:,corner_mask] = new_data

    with open(file_name,'r') as auto_file:

        header = auto_file.read().split('\n')
        header = '\n'.join(header[:2])

    autopilot_data[:,10] = auto_data[1]
    autopilot_data[:,12] = auto_data[2]

    autopilot_data[:,16] = calc_steering_angle(auto_data)

    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_cornerModel_original.csv", autopilot_data, delimiter=',', newline='\n', header=header, fmt="%.5f")


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
    auto = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',', dtype=float).T

    # consts = fit_corner_model(auto)

    consts = np.genfromtxt("trackData\\autopilot\\cornerModel_constants.csv", delimiter=",")

    produce_path("trackData\\testPaths\\2022-07-11 04.07.43 UTC 05.08.327 ideal.csv", track_name, consts, auto)

if __name__ == "__main__":

    main("nordschleife")



