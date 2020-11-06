import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, height, center, width, offset):
    return height * np.exp(-(x - center) ** 2 / (2 * width ** 2)) + offset

files = [ "../../01_simulated/03_final_results/00_interpolate_00_original.csv",
          "../../01_simulated/03_final_results/00_interpolate_01_camera.csv",
          "../../01_simulated/03_final_results/00_interpolate_02_distortion.csv",
          "../../01_simulated/03_final_results/01_homography_00_original.csv",
          "../../01_simulated/03_final_results/01_homography_01_camera.csv",
          "../../01_simulated/03_final_results/01_homography_02_distortion.csv" ]

titles = ["Polynomial (Original)", "Polynomial (Camera)", "Polynomial (Distortion)",
          "Homography (Original)", "Homography (Camera)", "Homography (Distortion)"]

for i, file in enumerate(files):
        plt.subplot(2, 3, i + 1)
        data = pd.read_csv(file)

        x = np.linspace(-200, 200, 21)
        y = data.groupby("camera_pos_x").error_deg_xyz.apply(lambda c: c.abs().mean()).values
        yerr = data.groupby("camera_pos_x").error_deg_xyz.apply(lambda c: c.abs().std()).values

        container = plt.bar(x, y, width=20, color="0.5", alpha=0.5, edgecolor="k", yerr=yerr, capsize=3)

        p1, cov_matrix = curve_fit(gaussian, x, y)
        x_plot = np.linspace(-210, 210, 1000)
        plt.plot(x_plot, gaussian(x_plot, *p1), 'r-', lw=2)

        plt.title(titles[i])

plt.show()
