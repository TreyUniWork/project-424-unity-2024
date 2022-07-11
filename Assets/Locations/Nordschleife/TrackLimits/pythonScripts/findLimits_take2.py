import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate

def loadTrackData(track_files, autopilot_data_filename):

    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)
    track, counts = np.unique(track, axis=0, return_counts=True)

    return track[counts<=4], None

def findTrackLimits(track, track_name):
    
    sideR = [2,1]; sideL = [40,42]
    sides = []
    track_vertices = track[:,[0,2]]
    next_point = -1

    # while point does not = this point & no points remaining

    for i, side in enumerate([sideL, sideR]):
        while next_point != side[0]:
            # find all points within 3 metres, if less, try again

            next_point = find_next_point(track_vertices, i, side)

            side.append(next_point)

        sides.append(side)

    return sides

def find_next_point(track, side, previous):
    # find next point in track limits

    prev = previous[-1]
    first = previous[0]

    dists = np.sum(np.square(track - track[prev]), axis=1)

    foundFirst = False

    n = 20
    dists[previous[1:]] = np.inf
    poss_points = np.argsort(dists)[1:1+n]

    if first in poss_points:
        first_ind = np.argwhere((poss_points == first))
        foundFirst = True

    b = track[poss_points] - track[prev]
    b = b.flatten()

    del_x = track[prev, 0] - track[previous[-2], 0]
    del_y = track[prev, 1] - track[previous[-2], 1]

    mag = np.sqrt(del_x**2 + del_y**2)
    parr = np.array([del_x, del_y]) / mag
    perp = np.array([-del_y, del_x]) / mag

    a = np.zeros((2*n,2*n))

    for i in range(n):
        a[i*2:i*2+2, i*2] = parr
        a[i*2:i*2+2, i*2+1] = perp

    arr = np.linalg.solve(a, b)
    parr = arr[::2]
    perp = arr[1::2]

    # magnitude = parr**2 + perp**2

    angle = np.arctan(perp/parr) * (1-2*side)

    angle[parr<0] = -np.inf

    next_cand = np.argmax(angle)

    ## find close neighbours

    # parr[angle < angle[next_cand] - 0.01] = np.inf
    # parr[parr < 0] = np.inf

    # next_cand = np.argmin(parr)

    # sns.set_theme()

    # fig = plt.figure()
    # ax = plt.axes()

    # ax.scatter(*track[poss_points].T, color = 'blue')
    # ax.scatter(*track[poss_points[next_cand]].T, color = 'green')
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
    
    # temp = track[poss_points]

    if foundFirst and parr[first_ind] > 0 and parr[first_ind] != np.inf:
        return first
    else:
        return poss_points[next_cand]

def plotTrack(*args):

    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

    for i, arg in enumerate(args):

        ax.scatter(arg[:,0], arg[:,2], color = colors[i % len(colors)])

    plt.title("Nordschleife")
    plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show()

def saveTrack(sides, track_name):

    with open(track_name+"_sides.txt", 'w') as sideFile:

        sideFile.write(','.join([str(x) for x in sides[0]])+'\n' + ','.join([str(x) for x in sides[1]]))
        sideFile.close()

def loadTrack(track_name):

    with open(track_name+"_sides.txt", 'r') as sideFile:
        
        sides = sideFile.read().split('\n')

        sideFile.close()

    return [[int(r) for r in x.split(',')] for x in sides]
    
def spline_curvature(tck):
    
    u = np.linspace(0, 1, int(1e5))
    midp = interpolate.splev(u, tck)
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    K = abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))

    return K, np.array(midp)

def interpolateTrack(track, sides):

    track_vertices = track[:, [0,2]]

    first = sides[0][0]

    dists = np.sum(np.square(track_vertices[sides[1]] - track_vertices[first]), axis=1)

    closest = np.argmin(dists)

    sides[1] = sides[1][:-1]
    sides[1] = sides[1][closest:] + sides[1][:closest] + [sides[1][closest]]

    tck_L, disc = interpolate.splprep(track_vertices[sides[0]].T)
    tck_R, disc = interpolate.splprep(track_vertices[sides[1]].T)

    new_u = np.linspace(0,1,int(1e5))

    sideL = interpolate.splev(new_u, tck_L) 
    sideR = interpolate.splev(new_u, tck_R)

    return np.array(sideL), np.array(sideR)


def main():

    track_name = "nordschleife"
    track, autopilot = loadTrackData((2,9), track_name + " unfinished.csv")

    sides = findTrackLimits(track, track_name)

    saveTrack(sides, track_name)
    sides = loadTrack(track_name)

    sideL, sideR = interpolateTrack(track, sides)

    midpoints = (sideL + sideR) / 2

    tck_mid, disc = interpolate.splprep(midpoints)

    curv, midp = spline_curvature(tck_mid)

    all_data = np.zeros((int(1e5),7))

    all_data[:,:2] = sideL.T
    all_data[:,2:4] = sideR.T
    all_data[:,4:6] = midp.T
    all_data[:,6] = curv

    np.savetxt(track_name+"_trackData.csv", all_data, delimiter=',', newline='\n')

if __name__ == "__main__":
    main()