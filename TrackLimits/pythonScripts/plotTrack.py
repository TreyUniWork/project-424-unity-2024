import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def plot_autopilot(new_auto_data, old_auto_data, sides, title):
    """_summary_

    Args:
        autopilot_data (_type_): _description_
        sides (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    siz = int(np.max(new_auto_data[-1]))+1
    starts = np.zeros(siz, dtype=int);ends=np.zeros(siz,dtype=int)

    for i in range(siz):

        corner_mask = np.nonzero(new_auto_data[-1]==i)[0]
        starts[i] = int(corner_mask[0])
        ends[i] = int(corner_mask[-1])

    ax.plot(*new_auto_data[1:3,(new_auto_data[-1]==0)], "r--", label="Corners")
    ax.plot(*new_auto_data[1:3,:starts[0]], "b--", label="Straights")

    ax.plot(*old_auto_data[1:3,(new_auto_data[-1]==0)], color="orange", label="Recorded")

    for i in range(1,siz):

        ax.plot(*new_auto_data[1:3,(new_auto_data[-1]==i)], "r--")
        # ax.text(*new_auto_data[1:3,(new_auto_data[-1]==i)][:,0], "Corner "+str(i))
        ax.plot(*new_auto_data[1:3,ends[i-1]:starts[i]+2], "b--")
        ax.plot(*old_auto_data[1:3,(new_auto_data[-1]==i)], color="orange")

    ax.plot(*new_auto_data[1:3,ends[-1]:], "b-") 

    ax.plot(*sides[0][1:3], "k-", label="Limits")
    ax.plot(*sides[1][1:3], "k-")

    # plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_optimiser(times):

    inf_mask = (times >399)
    times[inf_mask] = np.max(times[~inf_mask])
    best_sol = [times[0]]
    best_sol_inds = [0]

    for i, t in enumerate(times):
        if t < best_sol[-1]:
            best_sol.append(best_sol[-1])
            best_sol.append(t)
            best_sol_inds.append(best_sol_inds[-1])
            best_sol_inds.append(i)

    best_sol_inds.append(len(times))
    best_sol.append(best_sol[-1])

    sns.set_theme()

    fig, ax = plt.subplots()

    ax.plot(np.arange(times.size)[~inf_mask], times[~inf_mask], "*", color="green", label="Feasible")
    ax.plot(best_sol_inds, best_sol, color="orange", label="Best Solution", linewidth=3)

    plt.xlabel("Iterations")
    plt.ylabel("Objective (s)")
    plt.legend()
    plt.show()

    pass

def plot_solution(u, throttle, brake, coords_new, coords_old, sides, corner_def):
    """_summary_

    Args:
        u (_type_): _description_
        throttle (_type_): _description_
        brake (_type_): _description_
        coords_new (_type_): _description_
        coords_old (_type_): _description_
        sides (_type_): _description_
    """
    sns.set_theme()

    fig, axes = plt.subplots(1,2)

    axes[0].plot(*coords_old, "b--", label="Recorded")
    axes[0].plot(*coords_new[:,(corner_def == 0)], "r--", label="Optimal")
    for i in range(1,np.max(corner_def).astype(int)-1):
        axes[0].plot(*coords_new[:,(corner_def == i)], "r--")

    axes[0].plot(*sides[0][1:3], "k-", label="Limits")
    axes[0].plot(*sides[1][1:3], "k-")

    axes[0].set_title("Path Solution")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")

    axes[1].plot(u,throttle, color="green", label="Throttle")
    axes[1].plot(u,brake, color="red", label = "Brake")

    axes[1].set_title("Optimal Inputs")
    axes[1].set_xlabel("u (-)")
    axes[1].set_ylabel("Pedal %")

    axes[0].legend()
    axes[1].legend()
    plt.show()

    pass

def plot_constBounds(corner_data, corner_def, sides, title):
    """_summary_

    Args:
        autopilot_data (_type_): _description_
        sides (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    siz = int(np.max(corner_def))+1
    starts = np.zeros(siz, dtype=int);ends=np.zeros(siz,dtype=int)

    for i in range(siz):

        corner_mask = np.nonzero(corner_def==i)
        starts[i] = int(corner_mask[0][0])
        ends[i] = int(corner_mask[0][-1])

    colors = ["green", "orange", "red"]
    labels = ["Lower", "Starting", "Upper"]
    for j, c_data in enumerate(corner_data):
        for i in range(siz):
            if j == 0:
                if i == siz-1:
                    ax.plot(*c_data[:,ends[-1]:], "b-")
                elif i == 0:
                    ax.plot(*c_data[:,:starts[0]], "b--")
                    
                else:
                    ax.plot(*c_data[:,ends[i-1]:starts[i]+2], "b--")

            if i==0:
                ax.plot(*c_data[:,starts[i]:ends[i]], linestyle="dashed",color=colors[j], label = labels[j])
            else:
                ax.plot(*c_data[:,starts[i]:ends[i]], linestyle="dashed", color=colors[j])

    ax.plot(*sides[0][1:3], "k-")
    ax.plot(*sides[1][1:3], "k-")

    # plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_cornerLength(corner_data, corner_def, cs, sides, title):
    """_summary_

    Args:
        autopilot_data (_type_): _description_
        sides (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    siz = int(np.max(corner_def))+1
    starts = np.zeros(siz, dtype=int);ends=np.zeros(siz,dtype=int)

    for i in range(siz):

        corner_mask = np.nonzero(corner_def==i)[0]
        starts[i] = int(corner_mask[0])
        ends[i] = int(corner_mask[-1])

    colors = ["red", "orange", "yellow", "green", "magenta", "purple", ""]
    for j, c_data in enumerate(corner_data):
        for i in range(siz):
            if j == 0:
                if i == siz-1:
                    ax.plot(*c_data[:,ends[-1]:], "b-")
                elif i == 0:
                    ax.plot(*c_data[:,:starts[0]], "b--")
                    
                else:
                    ax.plot(*c_data[:,ends[i-1]:starts[i]+2], "b--")

            if i==0:
                ax.plot(*c_data[:,starts[i]:ends[i]], linestyle="dashed",color=colors[j], label = "C="+str(round(cs[j], 2)))
            else:
                ax.plot(*c_data[:,starts[i]:ends[i]], linestyle="dashed", color=colors[j])

    ax.plot(*sides[0][1:3], "k-")
    ax.plot(*sides[1][1:3], "k-")

    # plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_cornerSegments(auto_data, c_inds, sides, title):
    """_summary_

    Args:
        auto_data (_type_): _description_
        c_inds (_type_): _description_
        sides (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    siz = int(np.max(auto_data[-1]))+1
    starts = np.zeros(siz, dtype=int);ends=np.zeros(siz,dtype=int)

    for i in range(siz):

        corner_mask = np.nonzero(auto_data[-1]==i)[0]
        starts[i] = int(corner_mask[0])
        ends[i] = int(corner_mask[-1])


    ax.plot(*auto_data[1:3,:starts[0]], "b--", label="Straights")

    ax.plot(*auto_data[1:3,c_inds[0,0]:c_inds[0,1]+1], "r-", label="Entry Clothoid")
    ax.plot(*auto_data[1:3,c_inds[0,1]:c_inds[0,2]+1], "g-", label="Arc")
    ax.plot(*auto_data[1:3,c_inds[0,2]:c_inds[0,3]+1], "y-", label="Exit Clothoid")

    for i, c_i in enumerate(c_inds[1:]):
        
        ax.plot(*auto_data[1:3,c_i[0]:c_i[1]], "r-")
        ax.plot(*auto_data[1:3,c_i[1]:c_i[2]], "g-")
        ax.plot(*auto_data[1:3,c_i[2]:c_i[3]], "y-")

        ax.plot(*auto_data[1:3,ends[i-1]:starts[i]+2], "b--")

    ax.plot(*auto_data[1:3,ends[-1]:], "b-") 

    ax.plot(*sides[0][1:3], "k-", label="Limits")
    ax.plot(*sides[1][1:3], "k-")

    # plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_curvature(u, curvature, threshold, title):
    """_summary_

    Args:
        u (_type_): _description_
        curvature (_type_): _description_
        threshold (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()


    ax.plot(u, curvature, "r-")
    ax.plot([0,1],[threshold,threshold], "k--")
    ax.plot([0,1],[-threshold,-threshold], "k--")

    # plt.title(title)
    plt.xlabel("U")
    plt.ylabel("Curvature (1/m)")
    plt.show()

    pass

def plot_steeringFit(curvature, theta, betas):
    """_summary_

    Args:
        curvature (_type_): _description_
        theta (_type_): _description_
        betas (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    ax.plot(curvature, theta, "ro", label = "Data", markersize=1)
    ax.plot(curvature, betas[0]+betas[1]*curvature, "k--", label="Model")

    plt.xlabel("Curvature (1/m)")
    plt.ylabel("Steering Angle")
    plt.legend()

    plt.show()
    pass

def plot_autodata(auto_data, sides, title):
    """_summary_

    Args:
        auto_data (_type_): _description_
        sides (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    siz = int(np.max(auto_data[-1]))+1
    starts = np.zeros(siz, dtype=int);ends=np.zeros(siz,dtype=int)

    for i in range(siz):

        corner_mask = np.nonzero(auto_data[-1]==i)[0]
        starts[i] = int(corner_mask[0])
        ends[i] = int(corner_mask[-1])

    ax.plot(*auto_data[1:3,(auto_data[-1]==0)], "r-", label="Corners")
    ax.plot(*auto_data[1:3,:starts[0]], "b-", label="Straights")
    ax.text(*auto_data[1:3,(auto_data[-1]==0)][:,0], "Corner "+str(0))

    for i in range(1,siz):

        ax.plot(*auto_data[1:3,(auto_data[-1]==i)], "r-")
        ax.text(*auto_data[1:3,(auto_data[-1]==i)][:,0], "Corner "+str(i))
        ax.plot(*auto_data[1:3,ends[i-1]:starts[i]+2], "b-")

    ax.plot(*auto_data[1:3,ends[-1]:], "b-") 

    ax.plot(*sides[0][1:3], "k--", label="Limits")
    ax.plot(*sides[1][1:3], "k--")

    # plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_corner(corner_data, original_data, title):
    """_summary_

    Args:
        corner_data (_type_): _description_
        original_data (_type_): _description_
        num (_type_): _description_
        track_name (_type_): _description_
    """
    sns.set_theme()

    fig, axes = plt.subplots(nrows=2, ncols=2)

    # plt.suptitle(title)

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

    pass

def plot_defineCorner(u, curvature, start, end, new_start, new_end, peak_ind,k, title):
    """_summary_

    Args:
        u (_type_): _description_
        curvature (_type_): _description_
        start (_type_): _description_
        end (_type_): _description_
        peak_ind (_type_): _description_
        title (_type_): _description_
    """
    sns.set_theme()

    fig, ax = plt.subplots()

    ax.plot(u[start:new_start+1], curvature[start:new_start+1], "r-", label="Original")
    ax.plot(u[new_end:end], curvature[new_end:end], "r-")
    ax.plot(u[new_start:new_end+1], curvature[new_start:new_end+1], "b-", label="New")
    ax.plot(u[[start, peak_ind]], [curvature[start], curvature[peak_ind]*k], "k--", label="Min Gradient")
    ax.plot(u[[end, peak_ind]], [curvature[end], curvature[peak_ind]*k], "k--")

    plt.xlabel("U")
    plt.ylabel("Curvature (1/m)")
    plt.legend()
    plt.show()
    pass

def plot_limitFinding():
    """_summary_
    """
    # sns.set_theme()

    # fig = plt.figure()
    # ax = plt.axes()

    # ax.scatter(*track[poss_points].T, color = 'blue')
    # ax.scatter(*track[poss_points[angle!=-np.inf]].T, color = 'green')
    # vector_parr = track[prev] + [del_x, del_y]
    # vector_perp = track[prev] + [-del_y, del_x]

    # ax.plot([track[prev,0], vector_parr[0]], [track[prev,1], vector_parr[1]], color='orange', label = 'parallel')
    # ax.plot([track[prev,0], vector_perp[0]], [track[prev,1], vector_perp[1]], color='red', label = 'perpendicular')

    # plt.title("Nordschleife")
    # plt.tight_layout()
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.legend()
    # plt.show()

    pass
        
def plot_checkInside(coords, sides, title):
    """_summary_

    Args:
        corner_data (_type_): _description_
        sides (_type_): _description_
        title (_type_): _description_
    """


    sns.set_theme()

    fig, ax = plt.subplots()

    ax.plot(*coords, 'r--', label = "Corner")

    avg_pos = np.mean(coords, axis=1)

    dist1 = sides[0][1:3].T-avg_pos
    dist2 = sides[1][1:3].T-avg_pos

    dist1 = np.sum(np.square(dist1), axis=1)
    dist2 = np.sum(np.square(dist2), axis=1)

    closest1 = np.argmin(dist1)
    closest2 = np.argmin(dist2)

    ax.plot(*sides[0][1:3, closest1-20:closest1+20], 'k-', label="Sides")
    ax.plot(*sides[1][1:3, closest2-20:closest2+20], 'k-')

    
    # plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_interpolation(u,new_coords, old_coords, sides, title1, title2):
    """_summary_

    Args:
        new_data (_type_): _description_
        old_data (_type_): _description_
        sides (_type_): _description_
        title1 (_type_): _description_
        title2 (_type_): _description_
    """

    sns.set_theme()

    fig, ax = plt.subplots()

    error = np.sum(new_coords-old_coords, axis=0)
    ax.plot(u,error, 'r-')
    
    # plt.title(title2)
    plt.xlabel("U")
    plt.ylabel("Error (m)")

    plt.show()

    print(np.max(error))

    pass

def main():
    """_summary_
    """
    track_name = "nordschleife"
    corner_indices = np.genfromtxt("trackData\\testPaths\\"+track_name + "_autopilot_cornerIndices.csv", delimiter=',', dtype=int)
    autopilot_data = np.genfromtxt("trackData\\testPaths\\"+track_name + "_test.csv",  delimiter=',')
    new_auto_data = np.genfromtxt("trackData\\testPaths\\"+track_name + "_autopilot_autodata.csv", delimiter=',')
    old_auto_data = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',')
    sideL = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", delimiter=',', dtype=float)
    sideR = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", delimiter=',', dtype=float)
    sides = [sideL, sideR]

    # plot_autopilot(new_auto_data, old_auto_data, sides, "C")

    plot_cornerSegments(new_auto_data, corner_indices, sides, "")

if __name__ == "__main__":
    """_summary_
    """
    main()