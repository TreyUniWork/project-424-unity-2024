import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import interpolate

def loadTrackData(track_files):

    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("roadOutlines\\ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)
    track, counts = np.unique(track, axis=0, return_counts=True)

    return track[counts<=4]

def findTrackLimits(track):
    
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

def plotTrack(sides, auto, corners, mesh, mesh_sides):

    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(sides[:,0], sides[:,1], color = "black")
    ax.plot(sides[:,2], sides[:,3], color = "black", label = "Interp Sides")

    ax.scatter(*mesh[:,[0,2]].T, c="pink", s=1, label = "Original Mesh")

    ax.scatter(*mesh[mesh_sides[0]][:,[0,2]].T, c="purple", s=1, label = "Original Sides")
    ax.scatter(*mesh[mesh_sides[1]][:,[0,2]].T, c="purple", s=1)

    # for sideL in mesh_sides[0]:
    #     if mesh[sideL,0] > 1700 and mesh[sideL,0] < 1780 and mesh[sideL,2] > 210 and mesh[sideL,2] < 260:
    #         ax.text(*mesh[sideL][[0,2]].T,str(sideL))
    
    # for sideR in mesh_sides[1]:
    #     if mesh[sideR,0] > 1700 and mesh[sideR,0] < 1780 and mesh[sideR,2] > 210 and mesh[sideR,2] < 260:
    #         ax.text(*mesh[sideR][[0,2]].T,str(sideR))

    prev_end = 0
    for i, (start, end) in enumerate(corners):

        if i == 0:
                
            ax.plot(auto[prev_end:start,0], auto[prev_end:start,1], "b--", label="Straights")
            ax.plot(auto[start:end,0], auto[start:end,1], "g--", label = "Corners")
            prev_end = end
        else:
            ax.plot(auto[prev_end:start,0], auto[prev_end:start,1], "b--")
            ax.plot(auto[start:end,0], auto[start:end,1], "g--")
            prev_end = end

    plt.title("Nordschleife - Corner Splitting")
    # plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()

    plt.show()

def saveTrack(sides, track_name):

    with open("trackData\\"+track_name+"_sides.txt", 'w') as sideFile:

        sideFile.write(','.join([str(x) for x in sides[0]])+'\n' + ','.join([str(x) for x in sides[1]]))
        sideFile.close()

def loadTrack(track_name):

    with open("trackData\\"+track_name+"_sides.txt", 'r') as sideFile:
        
        sides = sideFile.read().split('\n')

        sideFile.close()

    return [[int(r) for r in x.split(',')] for x in sides]
    
def spline_curvature(tck, u):

    x, y = interpolate.splev(u, tck, der=0)
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    k = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))

    return np.array([u,x,y,dx,dy,k, u])

def interpolateTrack(track, sides):

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

    sideL = spline_curvature(tck_L)
    sideR = spline_curvature(tck_R)

    return [sideL,sideR]

def main():

    track_name = "nordschleife"
    track = loadTrackData((2,9))

    #sides = findTrackLimits(track)
    #saveTrack(sides, track_name)
    
    sides = loadTrack(track_name)

    sides = interpolateTrack(track, sides)

    np.savetxt("trackData\\"+track_name+"_trackData.csv", np.array(sides), delimiter=',', newline='\n')

def find_corners(curvature, threshold=3e-3):

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

    new_pairs = [[st, en] for st,en in zip(start,end) if en-st > 100]

    auto_corners = np.zeros(curvature.shape[0]) -1

    for i, (st,en) in enumerate(new_pairs):

        auto_corners[st:en] = i

    return auto_corners.astype(int)

def processAutopilot(file_name, track_name):

    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)

    with open(file_name,'r') as auto_file:

        header = auto_file.read().split('\n')
        header = '\n'.join(header[:2])

    tck_auto, u_points = interpolate.splprep(autopilot_data[:,[10,12]].T, w=autopilot_data[:,20])
    auto_data = spline_curvature(tck_auto, u_points)

    if True:
        new_coords = AutopilotSine(auto_data)
        autopilot_data[:,[10,12]] = new_coords.T

    auto_data[-1] = find_corners(auto_data[-1])

    np.savetxt("trackData\\testPaths\\"+track_name + "_autopilot_xyChange_wavy.csv", autopilot_data, delimiter=',', newline='\n', header=header)
    np.savetxt("trackData\\"+track_name + "_autopilot_interpolated.csv",auto_data, delimiter=',', newline='\n')


def AutopilotSine(auto_data, m=1, sigma=1e5):

    perp_vector = auto_data[[4,3]]
    mag = np.sqrt(np.sum(np.square(perp_vector), axis=0))
    perp_vector /= mag

    old_coord = auto_data[1:3]

    new_coord = old_coord + m * np.sin(2*np.pi*auto_data[0]*sigma) * perp_vector

    return new_coord
    

if __name__ == "__main__":
    #main()

    #x_dim*=-1

    processAutopilot("trackData\\testPaths\\2022-07-11 04.06.25 UTC 05.08.342.csv","nordschleife")

    #sides = loadTrack("nordschleife")

    #track = loadTrackData((2,9))