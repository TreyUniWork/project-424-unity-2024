import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def loadTrackData(track_files, autopilot_data_filename):

    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)
    track, counts = np.unique(track, axis=0, return_counts=True)

    return track[counts<=4], None

def plotTrack(*args):

    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

    for i, arg in enumerate(args):

        ax.scatter(arg[:100,0], arg[:100,2], color = colors[i % len(colors)], alpha=(np.arange(100)+1)/100)

    plt.title("Nordschleife")
    plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show()

def createClusters(track, track_name, s = 10):

    track_vertices = track[:,[0,2]]
    n = int(np.ceil(track_vertices.shape[0]/s))

    kmeans = KMeans(n_clusters = n, random_state = 0).fit(track_vertices)

    np.savetxt("Kmeans_labels_" + track_name + ".csv", kmeans.labels_, delimiter=',', newline='\n')
    np.savetxt("Kmeans_centers_" + track_name + ".csv", kmeans.cluster_centers_, delimiter=',', newline='\n')

def findTrackLimits(track, track_name, s = 40):
    
    track_vertices = track[:,[0,2]]

    n = int(np.ceil(track_vertices.shape[0]/(s)))

    labels, centers = findGroups(track_vertices, n, s)

    closest = closestTwo(centers, n)

    m = (centers[closest[:,1], 1] - centers[closest[:,0], 1]) / (centers[closest[:,1], 0] - centers[closest[:,0], 0])

    b = [track_vertices[labels==lab] - centers[lab] for lab in range(n)]

    a = np.zeros((2,2))
    z = np.zeros(track_vertices.shape[0])

    #not enough memory for a sparse array
    for i in range(n):

        #lets hope this works
        mag = np.sqrt(1+m[i] ** 2)
        parr = np.array([1, m[i]]) / mag
        perp = np.array([-m[i], 1]) / mag

        a[:,0] = parr
        a[:,1] = perp

        len_label = labels[labels==i].shape[0]

        temp_z = np.zeros(len_label)

        #find z distance
        for j in range(len_label):
            temp_z[j] = np.linalg.solve(a,b[i][j])[1]

        z[labels == i] = temp_z

    sides = np.zeros((n,2))

    #find positions of extreme points in group
    for i in range(n):

        masked_z = np.ma.array(z, mask=(labels!=i))

        sides[i] = [np.argmax(masked_z), np.argmin(masked_z)]

    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

    z_alpha = np.abs(z)
    for i in range(n):

        ax.scatter(*track_vertices[labels==i].T, s = 1e0, color = colors[i % len(colors)], label = str(i), alpha=z_alpha[labels==i] / np.max(z_alpha[labels==i]))
        ax.scatter(*centers[i], s = 10,color = colors[i % len(colors)])
        ax.plot(*centers[closest[i]].T, color = colors[i % len(colors)])

    plt.title("Nordschleife groups")
    plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    #plt.legend()
    plt.show()

    return track[sides.astype(int).T]

def closestTwo(centers, n):

    #find closest 2 clusters
    closest = np.zeros((n,2))

    for i in range(n):

        dists = np.sum(np.square(centers - centers[i]), axis=1)

        mask = np.zeros(n)
        mask[i] = 1

        dists_masked = np.ma.array(dists, mask=mask, fill_value=np.inf)
        closest1 = np.argmin(dists_masked)

        mask[closest1] = 1
        dists_masked = np.ma.array(dists, mask=mask, fill_value=np.inf)

        closest2 = np.argmin(dists_masked)

        closest[i] = [closest1, closest2]

    return closest.astype(int)

def findGroups(track, n, s):

    labels = np.zeros(track.shape[0], dtype=int) - 1

    indices = np.zeros(s, dtype=int)

    for i in range(n):
        
        #start with random

        compare_point = np.argmin(labels)

        dists = np.sum(np.square(track - track[compare_point]), axis=1)

        masked_dist = np.ma.array(dists, mask=(labels != -1), fill_value=np.inf)
        #assign most s's
        indices = np.argsort(masked_dist)[:int(s)]

        labels[indices] = i


    #attach to nearest centre

    centers = [np.mean(track[labels == lab], axis=0) for lab in range(n)]

    label_inds = np.arange(track.shape[0])[labels==-1]

    for i in zip(label_inds):

        dists = np.sum(np.square(centers - track[i]), axis=1)

        n_close = np.argmin(dists)

        labels[i] = n_close

    return labels, np.array(centers)


if __name__ == "__main__":

    track_name = "nordschleife"
    track, autopilot = loadTrackData((2,9), track_name + " unfinished.csv")

    #createClusters(track, track_name)

    #sides = findTrackLimits(track, track_name)

    plotTrack(track)