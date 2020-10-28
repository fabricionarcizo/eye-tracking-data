import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = glob.glob("../../01_simulated/02_data_analysis/01_homography/00_original/*.csv")
files.sort()

X = 21
Y = 21
Z = 21

errors_xy = np.zeros((X, Y))
errors_zy = np.zeros((Z, Y))

for i, file in enumerate(files):
    data = pd.read_csv(file)
    mean = data["error_deg_xyz"].values.mean()

    x = i // (X * Z)
    y = i // Z % Y
    z = i % Z

    errors_xy[x][y] += mean
    errors_zy[z][y] += mean


errors_xy /= Z
errors_zy /= Z

titles = [ "Influence of Camera Position (X- and Y-axes)",
           "Influence of Camera Position (Z- and Y-axes)" ]
axes = [ "Y-axis", "Z-axis" ]

for i in range(2):

    fig, ax = plt.subplots(1, 1)
    ax.set_title(titles[i])

    if i == 0:
        img = ax.imshow(errors_xy, cmap="binary", interpolation="bicubic")
    else:
        img = ax.imshow(errors_zy, cmap="binary", interpolation="bicubic")

    label_format = '{:,.0f}'
    if i == 0:
        ax.set_xticklabels([label_format.format(x) for x in np.linspace(-200, 200, 9)])
    else:
        ax.set_xticklabels([label_format.format(x) for x in np.linspace(-100, 700, 9)])

    ax.set_yticklabels([label_format.format(x) for x in np.linspace(12, 350, 10)])

    ax.set_xlabel("X-axis")
    ax.set_ylabel(axes[i])
    ax.invert_yaxis()

    fig.colorbar(img)

plt.show()

f = open("camera_movements_analysis.dat", "w")
for i, x in enumerate(np.linspace(-200, 200, X)):
    for j, y in enumerate(np.linspace(50, 350, Y)):
        mean = errors_xy[i][j]
        f.write("%d\t%d\t%f\n" % (x, y, mean))
    f.write("\n")
f.close()
