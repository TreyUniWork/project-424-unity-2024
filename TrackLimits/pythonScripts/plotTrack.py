
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

nums = [str(i) for i in range(2,3)]

segments = [np.genfromtxt("ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]

track = np.concatenate(segments, dtype=float)

sns.set_theme()

fig = plt.figure()
ax = plt.axes()
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

for i in range(len(segments)):

    ##seg = np.random.choice(segments[i].shape[0], size = 1000)
    ax.scatter(segments[i].T[0], segments[i].T[2], s=0.01)

plt.title("Nordschleife")
plt.tight_layout()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.show()


