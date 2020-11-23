# Imported libraries.
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import tee
from scipy import signal
from lmfit.models import GaussianModel

# Define pair of bins.
def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    left, right = tee(iterable)
    next(right, None)
    return zip(left, right)

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

outputs = [ "../../01_simulated/03_final_results/00_simulation_00_interpolate_00_original.csv",
            "../../01_simulated/03_final_results/00_simulation_00_interpolate_01_camera.csv",
            "../../01_simulated/03_final_results/00_simulation_00_interpolate_02_distortion.csv",
            "../../01_simulated/03_final_results/00_simulation_01_homography_00_original.csv",
            "../../01_simulated/03_final_results/00_simulation_01_homography_01_camera.csv",
            "../../01_simulated/03_final_results/00_simulation_01_homography_02_distortion.csv" ]

stats = [ "../../01_simulated/03_final_results/00_simulation_00_interpolate_00_original_sts.csv",
          "../../01_simulated/03_final_results/00_simulation_00_interpolate_01_camera_sts.csv",
          "../../01_simulated/03_final_results/00_simulation_00_interpolate_02_distortion_sts.csv",
          "../../01_simulated/03_final_results/00_simulation_01_homography_00_original_sts.csv",
          "../../01_simulated/03_final_results/00_simulation_01_homography_01_camera_sts.csv",
          "../../01_simulated/03_final_results/00_simulation_01_homography_02_distortion_sts.csv" ]

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
        errorX.append(data.error_deg_x.values)
        errorY.append(data.error_deg_y.values)
    errorX = np.asarray(errorX).reshape(-1)
    errorY = np.asarray(errorY).reshape(-1)

    # Calculate the X-axis histogram.
    hist, bins = np.histogram(errorX, bins=21)
    print(i, hist)

    # Calculate the histogram error.
    errors = []
    for left, right in pairwise(bins):
        if np.any((errorX >= left) & (errorX < right)):
            indices = np.where((errorX >= left) & (errorX < right))
            mean = errorX[indices].mean()
            std = errorX[indices].std()

            inside = np.where((errorX >= mean - std) & (errorX <= mean + std))
            error = len(errorX[inside]) / 2
        else:
            error = 0
        errors.append(error)
    errors = np.asarray(errors)

    # Histogram normalization.
    errors = errors / hist.sum()
    hist = hist / hist.sum()
    widths = bins[:-1] - bins[1:]

    # Calculate the Gaussian parameters.
    peaks, properties = signal.find_peaks(hist, width=True)

    if len(peaks) > 1:
        model = gauss1Model + gauss2Model
        p1, p2, p3 = np.unique((properties["left_bases"], properties["right_bases"]))
        params1 = gauss1Model.guess(hist[p1:p2], x=bins[p1:p2])
        params2 = gauss2Model.guess(hist[p2:p3], x=bins[p2:p3])
        params = params1.update(params2)
    else:
        model = gauss1Model
        params = model.guess(hist, x=bins[1:])

    # Fit the Gaussian curve.
    result = model.fit(hist, params, x=bins[1:])
    xPlot = np.linspace(bins[1:].min() * 1.10, bins[1:].max() * 1.10, 1000)
    # comps = result.eval_components(x=xPlot) // Evaluate the individual Gaussian component

    # Plot the histogram.
    plt.bar(bins[1:], hist, width=widths, edgecolor="k", alpha=0.35, color="k",
            yerr=errors, error_kw=dict(lw=1, capsize=2, capthick=1))
    plt.plot(xPlot, result.eval(x=xPlot), "k-", alpha=0.5, lw=2)
    # plt.axvline(errorX.mean(), color="k", alpha=0.5, linestyle='dashed', linewidth=1) # Plot the histogram mean

    # Plot the Gaussian components.
    # plt.plot(xPlot, comps["g1_"], "k--", alpha=0.25, lw=2)
    # if len(peaks) > 1:
    #     plt.plot(xPlot, comps["g2_"], "k--", alpha=0.25, lw=2)

    # Plot information.
    plt.title(titles[i])
    if i == 0 or i == 3:
        plt.ylabel("Normalized Density")
    if i > 2:
        plt.xlabel("Horizontal Gaze Error (deg)")
    plt.ylim([0, 0.325])

    # Saving the CSV data.
    output = np.vstack((bins[1:], hist, errors)).T
    df = pd.DataFrame(output, columns=[ "bins", "hist", "errors" ])
    df.to_csv(outputs[i], sep=",", index=False)

    output = np.vstack((xPlot, result.eval(x=xPlot))).T
    df = pd.DataFrame(output, columns=[ "x", "y" ])
    df.to_csv(stats[i], sep=",", index=False)

plt.show()
