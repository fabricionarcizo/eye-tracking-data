# <!--------------------------------------------------------------------------->
# <!--                           EyeInfo Framework                           -->
# <!--                        (http://eyeinfo.itu.dk)                        -->
# <!-- File       : plot_offset.py                                           -->
# <!-- Description: Plot the offset to show the accuracy and precision of    -->
# <!--            : eye tracking methods                                     -->
# <!-- Author     : Fabricio Batista Narcizo                                 -->
# <!--            : Rued Langgaards Vej 7, Kontor 4D25, 2300 København S.,   -->
# <!--            : Danmark                                                  -->
# <!--            : narcizo[at]itu[dot]dk                                    -->
# <!-- Responsible: Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)         -->
# <!--            : Fernando Dantas (feidantas2005[at]hotmail[dot]com)       -->
# <!--            : Copyright © 2019 ITU. All rights reserved.               -->
# <!-- Information: No additional information                                -->
# <!-- Date       : 22/11/2019                                               -->
# <!-- Changes    : 22/11/2019 - Create this file                            -->
# <!-- Review     : 22/11/2019 - Finalized                                   -->
# <!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019112201 $"


# ---------------------------------------------------------------------------- #
#                               Imported Headers                               #
# ---------------------------------------------------------------------------- #
import glob
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------- #
#                                    Script                                    #
# ---------------------------------------------------------------------------- #

# Search for a set of files that matches a specified pattern.
# filenames = glob.glob("../../01_simulated/03_final_results/*.csv")
filenames = glob.glob("../../02_real/03_final_results/*.csv")
filenames.sort()
stats = filenames[1::2]
filenames = filenames[::2]

# Titles.
titles = ["Polynomial (Original)", "Polynomial (Camera)", "Polynomial (Distortion)",
          "Homography (Original)", "Homography (Camera)", "Homography (Distortion)"]

# Create the Matplotlib figure.
fig = plt.figure()

# Process individually each file.
for i, stat, filename in zip(range(len(stats)), stats, filenames):

    # Define a new subplot.
    ax = fig.add_subplot(2, 3, i + 1)

    # Read the current CSV file.
    data = pd.read_csv(filename, sep=",")

    # Data structures to collect eye tracking data from the right ("\") and left
    # (".") eyes. I have used dictionaries to slipt the collect data into two
    # different groups.
    gaze = {}
    error_deg = {}

    for symbol in ["/", "."]:
        gaze[symbol] = []
        error_deg[symbol] = []

    # Read individually each eye tracking session (max. 166 experiments).
    for j in range(166):

        # Slice the eye tracking data based on the session ID.
        session = data.loc[data["experiment"] == j]

        # Get the result column name.
        column = "error_deg_xyz" if filenames[0].find("simulated") > -1 else "error_deg_xy"

        # I have excluded some sessions due to head movements and problems in
        # the eye feature extraction.
        if len(session) != 0:

            # Get the current eye data and calculate the mean error of all 35
            # gaze estimations.
            symbol = "/" if j % 2 == 0 else "."
            gaze[symbol].append(session[["error_deg_x", "error_deg_y"]].values[0])
            error_deg[symbol].append(session[column].values[0])

    # Convert vectors to numpy arrays.
    for symbol in ["/", "."]:
        gaze[symbol] = np.asarray(gaze[symbol])
        error_deg[symbol] = np.asarray(error_deg[symbol])

    # Read the current CSV file.
    data = pd.read_csv(stat, sep=",")

    # Calculate the mean and standard deviation of the current eye tracking
    # method.
    mean = np.vstack((gaze["/"], gaze["."])).mean(axis=0)
    std = data[["error_deg_x_std", "error_deg_y_std"]].values[0]

    # Calculate the plot limits.
    min_x, min_y = [mean[0] - std[0] * 4, mean[1] - std[1] * 4]
    max_x, max_y = [mean[0] + std[0] * 4, mean[1] + std[1] * 4]
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)    

    # Calculate and draw the dashed lines over the midpoint of eye tracking
    # distribution.
    ax.plot([mean[0], mean[0]], [min_y, max_y], "--", color="black")
    ax.plot([min_x, max_x], [mean[1], mean[1]], "--", color="black")

    # Calculate and draw the standard deviation.
    for j in range(1, 4):
        circle = Ellipse(mean, 2 * std[0] * j, 2 * std[1] * j, color="k", fill=False)
        ax.add_artist(circle)

    # Plot the gaze error.
    for symbol, label in zip(["/", "."], ["Right Eye", "Left Eye"]):
        ax.scatter(gaze[symbol][:, 0], gaze[symbol][:, 1], s=75, marker="o",
                    hatch=5 * symbol, label=label, alpha=0.5, facecolor="white",
                    edgecolors="black")

    # Plot information.
    ax.set_title(titles[i])
    if i == 0 or i == 3:
        ax.set_ylabel("Vertical Gaze Error (deg)")
    if i == 2:
        ax.legend(loc="upper right")
    if i > 2:
        ax.set_xlabel("Horizontal Gaze Error (deg)")

# Show the final plot.
plt.show()
