# <!--------------------------------------------------------------------------->
# <!--                   ITU - IT University of Copenhagen                   -->
# <!--                      Computer Science Department                      -->
# <!--                    Eye Information Research Group                     -->
# <!--       Introduction to Image Analysis and Machine Learning Course      -->
# <!-- File       : pccr_eye_feature.py                                      -->
# <!-- Description: Script to normalize the pupil center and calibration     -->
# <!--              targets based on PCCR method                             -->
# <!-- Author     : Fabricio Batista Narcizo                                 -->
# <!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
# <!--            : narcizo[at]itu[dot]dk                                    -->
# <!-- Responsible: Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)         -->
# <!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
# <!--              Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
# <!-- Information: No additional information                                -->
# <!-- Date       : 24/10/2018                                               -->
# <!-- Change     : 24/10/2018 - Creation of this script                     -->
# <!-- Review     : 24/10/2019 - Finalized                                   -->
# <!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019102401 $"

################################################################################
import cv2
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

################################################################################

# Get all the paths to the eye-videos.
files = glob.glob("./kde/0058*.csv")
files.sort()

# Define the normalized space.
square_unit = np.array([[-1, -1],
                        [+1, -1],
                        [+1, +1],
                        [-1, +1]])

# Define the glints available in all images.
glints_unit = [0,3]

# Process individually each video.
for file in files:

    # Read the csv file.
    data = pd.read_csv(file)

    # Define the output data set.
    keys = data.keys()[:4]
    df = pd.DataFrame(columns=keys)

    # Target coordinates.
    target_x = data["target_x"].to_numpy()
    target_y = data["target_y"].to_numpy()
    target = np.array([[target_x.min(), target_y.min()],
                       [target_x.max(), target_y.min()],
                       [target_x.max(), target_y.max()],
                       [target_x.min(), target_y.max()]])

    # Pupil coordinates.
    pupil_x = data["pupil_x"].to_numpy()
    pupil_y = data["pupil_y"].to_numpy()
    pupils = np.array([pupil_x, pupil_y, np.ones(pupil_x.shape[0])])

    # Homography normalization.
    H_t, _ = cv2.findHomography(target, square_unit)
    target = np.array([target_x, target_y, np.ones(target_x.shape[0])])
    target_norm = H_t.dot(target)

    # Process individually each eye feature.
    for i in range(35):

        # Normalize the glints 2 and 3.
        glints = data.loc[i][7:].to_numpy().reshape(4, 2)
        glint = glints[glints_unit].mean(axis=0)

        # PCCR normalization.
        pupil = pupils[:2, i] - glint[:2]

        # Save the normalized data.
        df.loc[i] = np.hstack((target[:2, i], pupil[:2]))

    # Set the filename.
    filename = "./normalized/" + file.split("/")[-1][:-4]
    df.to_csv(filename + ".csv", sep=",", encoding="utf-8", index=False)

    # Save the image.
    plt.clf()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(pupil_x, pupil_y)

    pupil_x = df["pupil_x"].to_numpy()
    pupil_y = df["pupil_y"].to_numpy()

    plt.subplot(1, 2, 2)
    plt.scatter(pupil_x, pupil_y)
    plt.savefig(filename + ".png", type="png", dpi=300)
