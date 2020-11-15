# Imported libraries.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from lmfit.models import GaussianModel

# Exponential and Gaussian model.
gauss1Model = GaussianModel(prefix="g1_")
gauss2Model = GaussianModel(prefix="g2_")

# Input CSV files.
files = [ "../../01_simulated/02_data_analysis/00_interpolate/00_original/*.csv",
          "../../01_simulated/02_data_analysis/00_interpolate/01_camera/*.csv",
          "../../01_simulated/02_data_analysis/00_interpolate/02_distortion/*.csv",
          "../../01_simulated/02_data_analysis/01_homography/00_original/*.csv",
          "../../01_simulated/02_data_analysis/01_homography/01_camera/*.csv",
          "../../01_simulated/02_data_analysis/01_homography/02_distortion/*.csv" ]

titles = [ "Polynomial (Original)", "Polynomial (Camera)", "Polynomial (Distortion)",
           "Homography (Original)", "Homography (Camera)", "Homography (Distortion)" ]

# Process each experiment individually.
for i, path in enumerate(files):
    plt.subplot(2, 3, i + 1)

    # Collect the eye-tracking data of each experiment.
    csvs = glob.glob(path)
    csvs.sort()

    errorX = []
    errorY = []
    for csv in csvs:
        data = pd.read_csv(csv)
        errorX.append(data.error_px_x.values)
        errorY.append(data.error_px_y.values)
    errorX = np.asarray(errorX).reshape(-1)
    errorY = np.asarray(errorY).reshape(-1)

    # Plot the X-axis error.
    hist, bins = np.histogram(errorX, bins=21, density=True)
    unityHist = hist / hist.sum()
    error = np.sqrt(hist) / hist.sum()
    widths = bins[:-1] - bins[1:]

    # Calculate the Gaussian parameters.
    peaks, properties = signal.find_peaks(unityHist, width=True)

    if len(peaks) > 1:
        model = gauss1Model + gauss2Model
        p1, p2, p3 = np.unique((properties["left_bases"], properties["right_bases"]))
        params1 = gauss1Model.guess(unityHist[p1:p2], x=bins[p1:p2])
        params2 = gauss2Model.guess(unityHist[p2:p3], x=bins[p2:p3])
        params = params1.update(params2)
    else:
        model = gauss1Model
        params = model.guess(unityHist, x=bins[1:])

    # Fit the Gaussian curve.
    print(i, bins)
    result = model.fit(unityHist, params, x=bins[1:])
    xPlot = np.linspace(bins.min() * 1.10, bins.max() * 1.10, 1000)
    comps = result.eval_components(x=xPlot)

    # Plot the histogram.
    plt.bar(bins[1:], unityHist, width=widths, edgecolor="k", alpha=0.5, color="k",
            yerr=error, error_kw=dict(lw=1, capsize=2, capthick=1))
    # plt.axvline(xPlot.mean(), color="k", alpha=0.5, linestyle='dashed', linewidth=1)

    plt.plot(xPlot, comps["g1_"], "k-", alpha=0.5, lw=2)
    if len(peaks) > 1:
        plt.plot(xPlot, comps["g2_"], "k-", alpha=0.5, lw=2)

    # Plot information.
    plt.title(titles[i])
    if i == 0 or i == 3:
        plt.ylabel("Normalized Density")
    if i > 2:
        plt.xlabel("Horizontal Gaze Error (metros)")

plt.show()
