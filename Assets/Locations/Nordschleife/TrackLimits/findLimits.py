import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def loadTrackData(track_files, autopilot_data_filename):

    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)
    track = np.unique(track, axis=0)

    return track, None

def plotTrack(*args):

    sns.set_theme()

    fig = plt.figure()
    ax = plt.axes()
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

    for i, arg in enumerate(args):

        ax.scatter(arg[:100,0], arg[:100,2], color = colors[i % len(colors)], s=1e0)

    plt.title("Nordschleife")
    plt.tight_layout()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show()


def createClusters(track, track_name, s = 20):

    track_vertices = track[:,[0,2]]
    n = int(np.ceil(track_vertices.shape[0]/s))

    kmeans = KMeans(n_clusters = n, random_state = 0).fit(track_vertices)

    np.savetxt("Kmeans_labels_" + track_name + ".csv", kmeans.labels_, delimiter=',', newline='\n')
    np.savetxt("Kmeans_centers_" + track_name + ".csv", kmeans.cluster_centers_, delimiter=',', newline='\n')

def findTrackLimits(track, track_name):
    
    track_vertices = track[:,[0,2]]

    labels = np.genfromtxt("Kmeans_labels_" + track_name + ".csv", delimiter=',', dtype=int)
    centers = np.genfromtxt("Kmeans_centers_" + track_name + ".csv", delimiter=',')

    n = centers.shape[0]

    closest = closestTwo(centers, n).astype(int)

    m = (centers[closest[:,0],1] - centers[closest[:,1],1]) / (centers[closest[:,0],0] - centers[closest[:,1],0])

    b = [track_vertices[labels==lab] - centers[lab] for lab in range(n)]
    b = np.array([col for row in b for col in row])

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

        len_label = labels[labels==i].shape[0]

        temp_z = np.zeros(len_label)
        temp_b = b[labels == i]

        #find z distance
        for j in range(len_label):
            temp_z[j] = np.linalg.solve(a,temp_b[j])[1]

        z[labels == i] = temp_z

    sides = np.zeros((n,2))

    #find positions of extreme points in group
    for i in range(n):

        masked_z1 = np.ma.array(z, mask=(labels!=i), fill_value=np.inf)
        masked_z2 = np.ma.array(z, mask=(labels!=i), fill_value=-np.inf)

        sides[i] = [np.argmin(masked_z1), np.argmax(masked_z2)]

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

    return closest


if __name__ == "__main__":

    track_name = "nordschleife"
    track, autopilot = loadTrackData((2,9), track_name + " unfinished.csv")

    # createClusters(track, track_name)

    sides = findTrackLimits(track, track_name)

    plotTrack(*sides)