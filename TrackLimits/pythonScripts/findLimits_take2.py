import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate

def loadTrackData(track_files):
    """_summary_

    Args:
        track_files (_type_): _description_

    Returns:
        _type_: _description_
    """
    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("roadOutlines\\ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)
    track, counts = np.unique(track, axis=0, return_counts=True)

    return track[counts<=4]

def findTrackLimits(track):
    """_summary_

    Args:
        track (_type_): _description_

    Returns:
        _type_: _description_
    """    
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

def find_next_point(track, side, previous, n=20):
    """_summary_

    Args:
        track (_type_): _description_
        side (_type_): _description_
        previous (_type_): _description_
        n (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """    
    # find next point in track limits

    prev = previous[-1]
    first = previous[0]

    dists = np.sum(np.square(track - track[prev]), axis=1)

    foundFirst = False

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

    angle[np.abs(angle)>np.pi/4] = -np.inf

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

    # if np.size(angle[angle > -1e3]) < 5:
    #     return find_next_point(track, side, previous, int((n+1)*1.1))
    # else:
    next_cand = np.argmax(angle)

    angle_diff = angle - angle[next_cand]

    angle[angle_diff < -(3*np.pi/180)] = -np.inf

    parr[angle == -np.inf] = np.inf

    next_cand = np.argmin(parr)

    if foundFirst and parr[first_ind] > 0 and parr[first_ind] != np.inf:
        return first
    else:
        return poss_points[next_cand]

def plotTrack(sideL, sideR, test_path):
    """_summary_

    Args:
        sideL (_type_): _description_
        sideR (_type_): _description_
        test_path (_type_): _description_
    """
    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(*sideL[1:3], color = "black")
    ax.plot(*sideR[1:3], color = "black")

    ax.plot(*test_path.T, color="orange")

    plt.title("Nordschleife - Corner Splitting")
    # plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    # plt.legend()

    plt.show()

def saveTrack(sides, track_name):
    """_summary_

    Args:
        sides (_type_): _description_
        track_name (_type_): _description_
    """
    with open("trackData\\"+track_name+"_sides.txt", 'w') as sideFile:

        sideFile.write(','.join([str(x) for x in sides[0]])+'\n' + ','.join([str(x) for x in sides[1]]))
        sideFile.close()

def loadTrack(track_name):
    """_summary_

    Args:
        track_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open("trackData\\"+track_name+"_sides.txt", 'r') as sideFile:
        
        sides = sideFile.read().split('\n')

        sideFile.close()

    return [[int(r) for r in x.split(',')] for x in sides]
    
def spline_curvature(tck, u, side = True):
    """_summary_

    Args:
        tck (_type_): _description_
        u (_type_): _description_
        side (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    x, y = interpolate.splev(u, tck, der=0)
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    k = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))

    psi = np.zeros(u.shape)

    mask_q1 = (dx>=0) & (dy>=0)
    mask_q2 = (dx<0) & (dy>=0)
    mask_q3 = (dx<0) & (dy<0)
    mask_q4 = (dx>=0) & (dy<0)

    psi[mask_q1] = np.arctan(dy[mask_q1]/dx[mask_q1])
    psi[mask_q2] = np.arctan(dy[mask_q2]/dx[mask_q2]) + np.pi
    psi[mask_q3] = np.arctan(dy[mask_q3]/dx[mask_q3]) + np.pi
    psi[mask_q4] = np.arctan(dy[mask_q4]/dx[mask_q4]) + np.pi*2

    mag = np.sqrt(dx**2+dy**2)

    if side:
        return np.array([u,x,y,psi, dx/mag, dy/mag])
    else:
        return np.array([u,x,y,k,psi,u])

def interpolateTrack(track, sides):
    """_summary_

    Args:
        track (_type_): _description_
        sides (_type_): _description_

    Returns:
        _type_: _description_
    """
    track[:,0] *= -1
    track_vertices = track[:, [0,2]]

    #find closest point
    first = sides[0][0]
    dists = np.sum(np.square(track_vertices[sides[1]] - track_vertices[first]), axis=1)
    closest = np.argmin(dists)

    #reset the loop
    sides[1] = sides[1][:-1]
    sides[1] = sides[1][closest:] + sides[1][:closest] + [sides[1][closest]]

    tck_L, disc = interpolate.splprep(track_vertices[sides[0]].T, s=1e1)
    tck_R, disc = interpolate.splprep(track_vertices[sides[1]].T, s=1e1)

    temp_u = np.linspace(0,1,int(1e4))

    sideL = spline_curvature(tck_L,temp_u)
    sideR = spline_curvature(tck_R, temp_u)

    return sideL,sideR

def main():
    """_summary_
    """
    track_name = "nordschleife"
    track = loadTrackData((2,9))

    #sides = findTrackLimits(track)
    #saveTrack(sides, track_name)
    
    sides = loadTrack(track_name)

    sides = interpolateTrack(track, sides)

    np.savetxt("trackData\\"+track_name+"_sidesDataL.csv", sides[0], delimiter=',', newline='\n')
    np.savetxt("trackData\\"+track_name+"_sidesDataR.csv", sides[1], delimiter=',', newline='\n')


def find_corners(curvature, threshold=3e-3):
    """_summary_

    Args:
        curvature (_type_): _description_
        threshold (_type_, optional): _description_. Defaults to 3e-3.

    Returns:
        _type_: _description_
    """
    auto_corners = np.zeros(curvature.shape[0])

    auto_corners[np.abs(curvature) > threshold] = 1

    mask = np.ediff1d(auto_corners)

    auto_corners[auto_corners == 0] = -1

    start = np.where((mask>0))[0] #  | ((auto_corners[1:] != -1) & (mask_change!=0))
    end = np.where((mask<0))[0] # | ((auto_corners[1:] == -1) & (mask_change!=0))

    if auto_corners[0] == 1:
        start = np.concatenate((np.array([0]), start))
    if auto_corners[-1] == 1:
        end = np.concatenate((end, np.array([-1])))

    new_pairs = [[st, en] for st,en in zip(start,end) if en-st > 50]

    auto_corners = np.zeros(curvature.shape[0]) -1

    for i, (st,en) in enumerate(new_pairs):

        auto_corners[st:en] = i

    return auto_corners.astype(int)

def processAutopilot(file_name, track_name):
    """_summary_

    Args:
        file_name (_type_): _description_
        track_name (_type_): _description_
    """
    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)

    tck_auto, u_points = interpolate.splprep(autopilot_data[:,[10,12]].T, w=autopilot_data[:,20])
    auto_data = spline_curvature(tck_auto, u_points, side=False)

    auto_data[-1] = find_corners(auto_data[3])

    np.savetxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv",auto_data.T, delimiter=',', newline='\n', fmt="%.5f")

def AutopilotSine(auto_data, m=0.5, sigma=5e2):
    """_summary_

    Args:
        auto_data (_type_): _description_
        m (float, optional): _description_. Defaults to 0.5.
        sigma (_type_, optional): _description_. Defaults to 5e2.

    Returns:
        _type_: _description_
    """

    b_vector = auto_data[1:3,1:] - auto_data[1:3,:-1]

    perp_vector = auto_data[[6,5]]

    steering_angle = np.zeros(auto_data.shape[1])

    for i, (p_vec, b_temp) in enumerate(zip(perp_vector.T,b_vector.T)):

        a = np.array([p_vec[::-1], p_vec])
        parr, perp = np.linalg.solve(a, b_temp)

        steering_angle[i] = np.arctan(perp/parr) * (180/np.pi)

    return steering_angle

if __name__ == "__main__":
    """_summary_
    """    
    #main()

    track_name = "nordschleife"

    processAutopilot("TrackLimits\\trackData\\testPaths\\initial_path.csv",track_name)

    #auto = np.genfromtxt("trackData\\autopilot"+track_name + "_autopilot_interpolated.csv", delimiter=',', dtype=float)
    # sideL = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", delimiter=',', dtype=float)
    # sideR = np.genfromtxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", delimiter=',', dtype=float)
    # test_path = np.genfromtxt("trackData\\testPaths\\"+track_name + "_autopilot_xyChange_wavy.csv", delimiter=',',skip_header=2, dtype=float)