# <!--------------------------------------------------------------------------->
# <!--                   ITU - IT University of Copenhagen                   -->
# <!--                      Computer Science Department                      -->
# <!--                    Eye Information Research Group                     -->
# <!--       Introduction to Image Analysis and Machine Learning Course      -->
# <!-- File       : normalize_eye_feature.py                                 -->
# <!-- Description: Script to normalize the pupil center and calibration     -->
# <!--              targets                                                  -->
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

def compute_rigid_transform(points_1, points_2):
    """Computes rotation, scale and translation for aligning points to unit."""

    A = np.array([[points_1[0, 0], -points_1[0, 1], 1, 0],
                  [points_1[0, 1], points_1[0, 0], 0, 1],
                  [points_1[1, 0], -points_1[1, 1], 1, 0],
                  [points_1[1, 1], points_1[1, 0], 0, 1]])
    y = points_2.reshape(1, 4)[0]

    # Least square solution to minimize ||Ax - y||.
    a, b, tx, ty = np.linalg.lstsq(A, y, rcond=None)[0]

    # Rotation matrix including scale.
    R = np.array([[a, -b], [b, a], [0, 0]])

    # Translation matrix.
    t = np.array([[tx], [ty], [1]])

    # Return the rigid transform matrix.
    return np.concatenate((R, t), axis=1)


# Get all the paths to the eye-videos.
files = glob.glob("./kde/*.csv")
files.sort()

# Define the normalized space.
square_unit = np.array([[-1, -1],
                        [+1, -1],
                        [+1, +1],
                        [-1, +1]])

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

    # Get the glint pattern.
    pattern = []
    for i in range(3, 35, 7):
        pattern.append(data.loc[i][7:])
    pattern = np.asarray(pattern)

    # Process individually each eye feature.
    for i in range(35):

        # Get the current glint pattern.
        glint_unit = np.hstack((pattern[i // 7].reshape(4, 2), np.ones((4, 1))))

        # Normalize the glints 2 and 3.
        glints = data.loc[i][7:].to_numpy().reshape(4, 2)
        if i % 7 < 2:
            glints = glints[1:3]
            M = compute_rigid_transform(glint_unit[1:3, :2], glints)
            glints = M.dot(glint_unit.T).T
        elif i % 7 > 4:
            glints = glints[0:4:3]
            M = compute_rigid_transform(glint_unit[0:4:3, :2], glints)
            glints = M.dot(glint_unit.T).T
        """elif i == 31:
            glints = glints[2:]
            M = compute_rigid_transform(glint_unit[2:, :2], glints)
            glints = M.dot(glint_unit.T).T"""

        # Homography normalization.
        H_t, _ = cv2.findHomography(glints, square_unit)
        if H_t is not None:
            pupil = H_t.dot(pupils[:, i])
            pupil /= pupil[2]
        else:
            pupil = np.array([-1, -1, 1])

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
    min_x, max_x = plt.xlim()
    plt.xlim(max_x, min_x)
    min_y, max_y = plt.ylim()
    plt.ylim(max_y, min_y)

    pupil_x = df["pupil_x"].to_numpy()
    pupil_y = df["pupil_y"].to_numpy()

    plt.subplot(1, 2, 2)
    plt.scatter(pupil_x, pupil_y)
    min_x, max_x = plt.xlim()
    plt.xlim(max_x, min_x)
    min_y, max_y = plt.ylim()
    plt.ylim(max_y, min_y)
    plt.savefig(filename + ".png", type="png", dpi=300)
