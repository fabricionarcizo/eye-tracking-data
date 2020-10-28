# <!--------------------------------------------------------------------------->
# <!--                   ITU - IT University of Copenhagen                   -->
# <!--                   SSS - Software and Systems Section                  -->
# <!--                     EyeInfo - Eye Information Lab                     -->
# <!-- File       : create_kde_data.py                                       -->
# <!-- Description: A script to calculate the kernel density estimation      -->
# <!--              (KDE) from a set of points                               -->
# <!-- Author     : Fabricio Batista Narcizo                                 -->
# <!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
# <!--            : narcizo[at]itu[dot]dk                                    -->
# <!-- Responsible: Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)         -->
# <!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
# <!--              Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
# <!-- Information: No additional information                                -->
# <!-- Date       : 23/09/2019                                               -->
# <!-- Change     : 23/09/2019 - Creation of this class                      -->
# <!-- Review     : 22/10/2019 - Finalized                                   -->
# <!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019102201 $"

################################################################################
import cv2
import os
import numpy as np
import pandas as pd

from scipy import stats


################################################################################

# ---------------------------------------------------------------------------- #
#                               Script Functions                               #
# ---------------------------------------------------------------------------- #

# <!--------------------------------------------------------------------------->
# <!-- Function   : def calculate_kde()                                      -->
# <!-- Arguments  : feature                                                  -->
# <!-- Return     : kde                                                      -->
# <!-- Information: Calculate the eye feature as the center of KDE           -->
# <!--------------------------------------------------------------------------->
def calculate_kde(feature):
    """Calculate the eye feature as the center of KDE."""

    # Removes the invalid elements.
    feature = feature[~np.all(feature[:, 0:2] == -1, axis=1)]

    # Removes the outliers in the eye feature distribution.
    if feature.shape[0] > 4:
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        feature = feature[~np.any(abs(feature - mean) > std, axis=1)]

    # Check if there are valid eye feature.
    if feature.shape[0] == 0:
        return np.array([-1, -1], np.float64)

    # Verify the midpoint and the standard deviation.
    if feature.shape[0] == 0:
        return np.array([-1, -1], np.float64)

    elif feature.shape[0] < 2:
        return np.array([-1, -1], np.float64)

    # Calculate the range of eye feature in both axes.
    m1, m2 = feature[:, 0], feature[:, 1]
    x_min, x_max = m1.min(), m1.max()
    y_min, y_max = m2.min(), m2.max()

    # Check if there are enough data.
    if x_min == x_max or y_min == y_max:
        return np.mean(feature, axis=0)

    # Calculate the Kernel Density Estimate (KDE) on the data.
    x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([x.ravel(), y.ravel()])
    values = np.vstack([m1, m2])

    # Check if there are enough data.
    if values.shape[1] < 3:
        return values.mean(axis=1)

    # Applies the Kernel Density Estimation based on Gaussian curve.
    try:
        kernel = stats.gaussian_kde(values)
        kde = np.reshape(kernel(positions).T, x.shape)
        kde = kde * (10 ** 5)
    except:
        return values.mean(axis=1)

    # Get the X- and Y- coordinates of maximum KDE distribution.
    i, j = np.unravel_index(kde.argmax(), kde.shape)

    # Return the center of KDE.
    return np.array([x[i, j], y[i, j]], dtype=np.float64)


###############################################################################

# Process all data set files.
for i in range(166):

    # Get the folder name.
    folder = "%04d" % i

    # Get the data set path.
    filename = "./datasets/%s_dataset.csv" % folder

    # Read the csv file.
    data = pd.read_csv(filename)

    # Define the output data set.
    keys = data.keys()[3:]
    df = pd.DataFrame(columns=keys)

    # Analyze individually each calibration target.
    for j in range(35):

        # Eye-tracking calibration data.
        calibration_data = data.loc[(data["target_no"] == j)]

        # Target coordinates.
        target_x = calibration_data["target_x"].to_numpy()[0]
        target_y = calibration_data["target_y"].to_numpy()[0]

        # Pupil coordinates.
        pupil_x = calibration_data["pupil_x"].to_numpy()
        pupil_y = calibration_data["pupil_y"].to_numpy()

        # Pupil center distribution.
        pupils = calculate_kde(np.vstack((pupil_x, pupil_y)).T)

        # Ellipse axes.
        ellipse_min_axis = calibration_data["ellipse_min_axis"].to_numpy()
        ellipse_max_axis = calibration_data["ellipse_max_axis"].to_numpy()

        # Ellipse axes distribution.
        ellipse_axes = calculate_kde(np.vstack((ellipse_min_axis,
                                                ellipse_max_axis)).T)

        # Ellipse angle mean.
        ellipse_angle = calibration_data["ellipse_angle"].to_numpy().mean()

        # Analyze individually each glint.
        glints = []
        for k in range(1, 5):
            # Glints coordinates.
            glint_x = calibration_data["glint_%s_x" % k].to_numpy()
            glint_y = calibration_data["glint_%s_y" % k].to_numpy()

            # Glint center distribution.
            glints.append(calculate_kde(np.vstack((glint_x, glint_y)).T))

        # Add a new row.
        df.loc[j] = [target_x, target_y, pupils[0], pupils[1], ellipse_axes[0],
                     ellipse_axes[1], ellipse_angle, glints[0][0], glints[0][1],
                     glints[1][0], glints[1][1], glints[2][0], glints[2][1],
                     glints[3][0], glints[3][1]]

        # Open the mean image.
        filename = "./mean/%s/%s_%sx%s.png" % (folder, "%02d" % j,
                                               "%04d" % target_x,
                                               "%04d" % target_y)
        image = cv2.imread(filename)

        # Draw the eye feature in the image.
        if pupils[0] != -1 and pupils[1] != -1:
            cv2.circle(image, tuple(pupils.astype(int)), 5, (0, 255, 0), -1)
            ellipse = (tuple(pupils), tuple(ellipse_axes), ellipse_angle)
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)
        for k, glint in enumerate(glints):
            if glint[0] != 1 and glint[1] != -1:
                cv2.circle(image, tuple(glint.astype(int)),
                           5, (255, 0, 255), -1)
                image = cv2.putText(image, str(k + 1), tuple(glint.astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)

        # Create the output folder.
        if not os.path.exists("processed/%s" % folder):
            os.makedirs("processed/%s" % folder)

        # Save the processed image.
        filename = "./processed/%s/%s_%sx%s.png" % (folder, "%02d" % j,
                                                    "%04d" % target_x,
                                                    "%04d" % target_y)
        cv2.imwrite(filename, image)

    # Save the data frame.
    filename = "./kde/%s_dataset.csv" % folder
    df.to_csv(filename, sep=",", encoding="utf-8", index=False)
