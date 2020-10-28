import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

fontP = FontProperties()
fontP.set_size('xx-small')

# files = glob.glob("../../01_simulated/02_data_analysis/00_interpolate/00_original/*.csv")
# files = glob.glob("../../01_simulated/02_data_analysis/00_interpolate/01_camera/*.csv")
# files = glob.glob("../../01_simulated/02_data_analysis/00_interpolate/02_distortion/*.csv")
# files = glob.glob("../../01_simulated/02_data_analysis/01_homography/00_original/*.csv")
# files = glob.glob("../../01_simulated/02_data_analysis/01_homography/01_camera/*.csv")
files = glob.glob("../../01_simulated/02_data_analysis/01_homography/02_distortion/*.csv")
files.sort()

title = "Original Polynomial"
# title = "Polynomial + Camera Correction"
# title = "Polynomial + Camera and Distortion Corrections"
# title = "Original Homography"
# title = "Homography + Camera Correction"
title = "Homography + Camera and Distortion Corrections"

X = 21
Y = 21
Z = 21

errors = np.zeros((X * Y * Z, 4))
distances = np.zeros((X * Y * Z))

f = open("camera_movements_analysis_distortion.dat", "w")

for i, file in enumerate(files):
    data = pd.read_csv(file)
    mean = data['error_deg_xyz'].values.mean()

    x = int(file.split("/")[-1].split("_")[1][1:])
    y = int(file.split("/")[-1].split("_")[2][1:])
    z = int(file.split("/")[-1].split("_")[3][1:])

    errors[i][0] = x
    errors[i][1] = y
    errors[i][2] = z
    errors[i][3] = mean

    distances[i] = mean

    a = i // (X * Z)
    b = i // Z % Y
    c = i % Z

    if c == 0 or c == 10 or c == 20:
        f.write("%d,%d,%d,%f\n" % (x, y, z, mean))

f.close()
print("%f\t%f\n" % (distances.min(), distances.max()))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
scatter = ax.scatter(
    xs=errors[:, 0],
    ys=errors[:, 2],
    zs=errors[:, 1],
    c=distances,
    cmap='binary',
    vmin=-1.0, vmax=2.5
)

ax.set_title(title)
ax.set_xlim(errors[:, 0].min()-25, errors[:, 0].max()+25)
ax.set_ylim(errors[:, 2].max()+25, errors[:, 2].min()-25)
ax.set_zlim(errors[:, 1].min()-25, errors[:, 1].max()+25)
ax.set_xlabel('X-axis')
ax.set_ylabel('Z-axis')
ax.set_zlabel('Y-axis')

handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, labels, title='Gaze Estimations', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.show()

#tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)

"""tsne = PCA(n_components=3)
tsne_results = tsne.fit_transform(errors)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
scatter = ax.scatter(
    xs=tsne_results[:, 0],
    ys=tsne_results[:, 1],
    zs=tsne_results[:, 2],
    c=distances,
    cmap='tab10'
)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

tsne = PCA(n_components=2)
tsne_results = tsne.fit_transform(errors)

scatter = plt.scatter(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    c=distances,
    cmap='binary'
)

handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, labels, title='Gaze Estimations', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
plt.show()
"""
