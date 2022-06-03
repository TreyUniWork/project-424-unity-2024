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

def calcDist(p1, p2):

    dist = (p2-p1)**2

    return np.sum(dist)

def findTrackLimits(track, s = 10):
    
    track_vertices = track[:,[0,2]]
    shape = track_vertices.shape[0]

    n = int(np.ceil(shape/s))

    kmeans = KMeans(nclusters = n, random_state = 0).Fit(track_vertices)

    #find closest 2 clusters
    closest = np.zeros(n,2)
    for i in range(n):
        max_dist1 = np.inf
        max_dist2 = np.inf
        max_1 = i-1
        max_2 = i+1

        p1 = kmeans.cluster_centers_[i]
        for j in range(n):
            if i == j:
                continue
            p2 = kmeans.cluster_centers_[j]

            dist = calcDist(p1,p2)

            if dist < max_dist1:
                max_dist2 = max_dist1
                max_dist1 = dist
                max_2 = max_1
                max_1 = j
            elif dist < max_dist2:
                max_dist2 = dist
                max_2 = j
        
        closest[i] = [max_1, max_2]

    m = (kmeans.cluster_centers_[closest[:,0],1] - kmeans.cluster_centers_[closest[:,1],1]) / (kmeans.cluster_centers_[closest[:,0],0] - kmeans.cluster_centers_[closest[:,1],0])

    b = track_vertices[kmeans.labels_] - kmeans.cluster_centers_

    a = np.zeros((2,2))
    z = np.zeros(track_vertices.shape[0])

    #not enough memory for a sparse array
    for i in range(n):

        #lets hope this works
        mag = np.sqrt(1+m[i]**2)
        parr = np.array([m[i], 1]) / mag
        perp = np.array([1, -m[i]]) / mag

        a[:,0] = parr[::-1]
        a[:,1] = perp[::-1]

        temp_z = z[kmeans.labels_ == i]

        #find z distance
        for j in len(kmeans.labels_ == i):
            temp_z[j] = np.linalg.solve(a,b[j])[1]

        z[kmeans.labels_ == i] = temp_z

    sides = np.zeros((n,2))

    #find positions of extreme points in group
    for i in range(n):

        masked_z1 = np.ma.array(z, mask=(kmeans.labels_==i), fill_value=np.inf)
        masked_z2 = np.ma.array(z, mask=(kmeans.labels_==i), fill_value=-np.inf)

        sides[n] = [np.argmin(masked_z1), np.argmax(masked_z2)]

    return track[sides.T]


if __name__ == "__main__":

    track, autopilot = loadTrackData((2,9), "nordschleife" + " unfinished.csv")

    side1, side2 = findTrackLimits(track)

    plotTrack(side1, side2)