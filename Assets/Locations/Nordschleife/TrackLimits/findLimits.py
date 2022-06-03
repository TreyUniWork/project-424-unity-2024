import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def loadTrackData(track_files, autopilot_data_filename):

    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)

    return track, None

def plotTrack(*args):

    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

    for i, arg in enumerate(args):

        ax.scatter(arg[:,0], arg[:,2], color = colors[i % len(colors)], s=1e0)

    plt.title("Nordschleife")
    plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show()

    pass

def findTrackLimits(track, s = 10):
    
    track_vertices = track[:,[0,2]]
    shape = track_vertices.shape[0]

    n = int(np.ceil(shape/s))

    #obsolete with k-means
    # inds = np.arange(n)*s
    # inds = np.append(inds, track_vertices.shape[0])

    # obsolete with K-means
    # avg_points = [np.mean(track_vertices[start:end], axis=0) for start, end in zip(inds[:-1], inds[1:])]
    # avg_points = np.array(avg_points)

    ii = [0] + [i for i in range(n-1)]
    jj = [i for i in range(1,n)] + [n-1]
    ii = np.array(ii); jj = np.array(jj)

    m = (avg_points[ii,1] - avg_points[jj,1]) / (avg_points[ii,0] - avg_points[jj,0])

    b = [track_vertices[start:end] - avg_points[i] for i, (start, end) in enumerate(zip(inds[:-1], inds[1:]))]
    b = np.array([coord for bco in b for coord in bco])

    a = np.zeros((2,2))
    z = np.zeros(b.shape[0])

    #not enough memory for a sparse array
    for i, (start, end) in enumerate(zip(inds[:-1], inds[1:])):

        #lets hope this works
        mag = np.sqrt(1+m[i]**2)
        parr = np.array([m[i], 1]) / mag
        perp = np.array([1, -m[i]]) / mag

        a[:,0] = parr[::-1]
        a[:,1] = perp[::-1]

        for j in range(start, end):

            z[j] = np.linalg.solve(a,b[j])[1]

    sides = [[start + np.argmin(z[start:end]), start+np.argmax(z[start:end])] for start, end in zip(inds[:-1], inds[1:])]
    sides = np.array(sides).T

    return track[sides]

if __name__ == "__main__":


    track, autopilot = loadTrackData((2,9), "nordschleife" + " unfinished.csv")

    track_ordered = orderTrack(track)

    side1, side2 = findTrackLimits(track_ordered)

    plotTrack(side1, side2)