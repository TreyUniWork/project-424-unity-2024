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

    ax.plot(*new_auto_data[1:3,(new_auto_data[-1]==0)], "r-", label="Corners")
    ax.plot(*new_auto_data[1:3,:starts[0]], "b-", label="Straights")

    ax.plot(*old_auto_data[1:3], color="orange", label="Recorded")

    for i in range(1,siz):

        ax.plot(*new_auto_data[1:3,(new_auto_data[-1]==i)], "r-")
        ax.text(*new_auto_data[1:3,(new_auto_data[-1]==i)][:,0], "Corner "+str(i))
        ax.plot(*new_auto_data[1:3,ends[i-1]:starts[i]+2], "b-")

    ax.plot(*new_auto_data[1:3,ends[-1]:], "b-") 

    ax.plot(*sides[0][1:3], "k--", label="Limits")
    ax.plot(*sides[1][1:3], "k--")

    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.legend()
    plt.show()

    pass

def plot_autodata(auto_data, sides, title):

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

    plt.title(title)
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

    plt.suptitle(title)

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

def plot_limitFinding():

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

def plot_cornerLength(corner_data, sides, title):

    pass
def plot_checkInside(corner_data, sides, title):

    pass

def main():

    track_name = "nordschleife"
    corner_indices = np.genfromtxt("trackData\\testPaths\\"+track_name + "_autopilot_cornerIndices.csv", delimiter=',')
    autopilot_data = np.genfromtxt("trackData\\testPaths\\"+track_name + "_test.csv",  delimiter=',')
    new_auto_data = np.genfromtxt("trackData\\testPaths\\"+track_name + "_autopilot_autodata.csv", delimiter=',')
    old_auto_data = np.genfromtxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv", delimiter=',')
    sideL = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", delimiter=',', dtype=float)
    sideR = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", delimiter=',', dtype=float)
    sides = [sideL, sideR]

    plot_autopilot(new_auto_data, old_auto_data, sides, "C")

if __name__ == "__main__":

    main()