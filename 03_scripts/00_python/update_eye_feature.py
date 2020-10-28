
#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : update_eye_feature.py                                    -->
#<!-- Description: Script to detect and update upils and glints in eye      -->
#<!--            : images using binary images and blob detection            -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : narcizo[at]itu[dot]dk                                    -->
#<!-- Responsable: Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!--              Zaheer Ahmed (zaah[at]itu[dot]dk)                        -->
#<!--              Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 06/11/2019                                               -->
#<!-- Change     : 06/11/2019 - Creation of this script                     -->
#<!-- Review     : 06/11/2019 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019110901 $"

########################################################################
import cv2
import math
import glob
import os
import numpy as np
import IAMLTools
import pandas as pd
from pandas import DataFrame

########################################################################

def onValuesChange(dummy=None):
    """ Handle updates when slides have changes."""
    global trackbarsValues, ellipses, centers, bestPupilID, current_id
    trackbarsValues["threshold"] = cv2.getTrackbarPos("threshold", "Threshold")
    trackbarsValues["minimum"]   = cv2.getTrackbarPos("minimum", "Threshold")
    trackbarsValues["maximum"]   = cv2.getTrackbarPos("maximum", "Threshold")
    trackbarsValues["glint"]     = cv2.getTrackbarPos("glint", "Threshold")

    ellipses, centers, bestPupilID = detectPupil(frame, trackbarsValues["threshold"],
                trackbarsValues["minimum"], trackbarsValues["maximum"])
    glints, bestGlintsID = detectGlints(frame, trackbarsValues["glint"])
    showDetectedPupil(frame, threshold, ellipses, centers, bestPupilID,
                      glints, bestGlintsID, current_id, trackbarsValues[
                          "glint"])


def click_and_select(event, x, y, flags, param):
    """ Select the glint ID by mouse clicks."""
    global current_id, coordinates

    if event == cv2.EVENT_LBUTTONDOWN:
        if (current_id is not None):
            coordinates[current_id - 1] = (x * 2, y * 2)
        current_id = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        coordinates = [(-1, -1)] * 4

    onValuesChange()


def showDetectedPupil(image, threshold, ellipses=None, centers=None, bestPupilID=None,
                      glints=None, bestGlintsID=None, current_id=None,
                      glint_thres=None):
    """"
    Given an image and some eye feature coordinates, show the processed image.
    """
    global is_first, target_id, folder

    # Copy the input image.
    processed = image.copy()
    if (len(processed.shape) == 2):
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale.
    grayscale = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Create a binary image using manual threshold values.
    _, thresh = cv2.threshold(grayscale, glint_thres, 255, cv2.THRESH_BINARY)
    processed[thresh == 255] = [255, 0, 255]
    processed[thresh == 255] = [255, 0, 255]

    # Draw the eye feature.
    if (ellipses is not None and len(ellipses) > 0 and
        centers  is not None and len(centers) > 0):
        for pupil, center in zip(ellipses, centers):
            cv2.ellipse(processed, pupil, (0, 0, 255), 2)
            if center[0] != -1 and center[1] != -1:
                cv2.circle(processed, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)            

        # Draw the best pupil candidate:
        if (bestPupilID is not None and bestPupilID != -1):
            pupil  = ellipses[bestPupilID]
            center = centers[bestPupilID]


            drawCross(processed, (int(center[0]), int(center[1])), 0,
                      trackbarsValues["minimum"], (0, 255, 0))
            drawCross(processed, (int(center[0]), int(center[1])), 45,
                      trackbarsValues["maximum"], (0, 255, 0))

            cv2.ellipse(processed, pupil, (0, 255, 0), 2)
            if center[0] != -1 and center[1] != -1:
                cv2.circle(processed, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)

    if (bestGlintsID is not None):
        for pos, gid in enumerate(bestGlintsID):
            if (gid != -1):
                image = cv2.putText(processed, str(pos + 1),
                                    (int(glints[gid][0]), int(glints[gid][1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)

    # Show current glint.
    if (current_id is not None):
        image = cv2.putText(processed, "Glint: %d" % current_id, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

    # Save the image.
    if (is_first):

        if not os.path.exists("figs/%s" % folder):
            os.makedirs("figs/%s" % folder)

        cv2.imwrite("figs/%s/%s_target_%s.png" % (folder, folder, target_id), processed)

    # Show the processed image.
    tmp = processed.copy()
    tmp = cv2.resize(tmp, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Eye", tmp)


def drawCross(image, center, angle, distance, color):
    """Drawing a cross over a determined region in the processed image."""
    if np.mean(center) == 0:
        return
    for i in range(4):
        point = (int(center[0] + distance * np.cos(angle * np.pi / 180.0)),
                 int(center[1] + distance * np.sin(angle * np.pi / 180.0)))
        cv2.line(image, center, point, color, 2)
        angle = angle + 90


def detectPupil(image, threshold=0, minimum=10, maximum=50, 
                kernel_size=9):
    """
    Given an image, return the coordinates of the pupil candidates.
    """
    # Create the output variable.
    bestPupilID = -1
    ellipses    = []
    centers     = []

    # Grayscale image.
    grayscale = image.copy()
    if len(grayscale.shape) == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)
        
    # Add gaussian blur
    grayscale = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    # Define the minimum and maximum size of the detected blob.
    min_ellipse = minimum
    max_ellipse = maximum
    minimum = int(round(math.pi * math.pow(minimum, 2)))
    maximum = int(round(math.pi * math.pow(maximum, 2)))

    # Create a binary image using manual threshold values.
    _, thresh = cv2.threshold(grayscale, threshold, 255,
                              cv2.THRESH_BINARY_INV)

    # Show the threshould image.
    tmp = thresh.copy()
    tmp = cv2.resize(tmp, (0, 0), fx=0.35, fy=0.35)
    cv2.imshow("Threshold", tmp)

    # Find blobs in the input image.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    #<!--------------------------------------------------------------------------->
    #<!--                            YOUR CODE HERE                             -->
    #<!--------------------------------------------------------------------------->
    
    # Best circularity.
    bestCircularity = 0

    # Analyze each detected blob.
    for contour in contours:
        # Get the contour properties.
        properties = IAMLTools.getContourProperties(contour, ["Area", "Centroid"])

        # Improve the contour by compugin the convex hull.
        contour = cv2.convexHull(contour)

        # Estimate the center of the blob.
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
        else:
            ellipse = cv2.minAreaRect(contour)

        # Get the blob parameters.
        area   = properties["Area"]
        center = properties["Centroid"]

        # Blob classification.
        if (area < minimum or area > maximum):
            continue

        # Aspect ratio.
        if min(ellipse[1]) / max(ellipse[1]) < 0.5:
            continue

        if (min(ellipse[1]) // 2 < min_ellipse or
            max(ellipse[1]) // 2 > max_ellipse):
            continue

        # Fill out the blob parameters vector.
        ellipses.append(ellipse)
        centers.append(center)

        # Evaluate the blob circularity.
        properties  = IAMLTools.getContourProperties(contour, ["Circularity"])
        circularity = properties["Circularity"]

        if (abs(1. - circularity) < abs(1. - bestCircularity)):
            bestCircularity = circularity
            bestPupilID = len(ellipses) - 1

    #<!--------------------------------------------------------------------------->
    #<!--                                                                       -->
    #<!--------------------------------------------------------------------------->

    # Return the final result.
    return ellipses, centers, bestPupilID


def detectGlints(image, glint_thres=150, kernel_size=9):
    '''
    Given an image, return the coordinates of the glint candidates.
    '''
    global coordinates

    # Create the output variable.
    bestGlintsIDS = [-1] * 4
    centers       = []

    # Grayscale image.
    grayscale = image.copy()
    if len(grayscale.shape) == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    # Add gaussian blur
    grayscale = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    # Create a binary image using manual threshold values.
    _, thresh = cv2.threshold(grayscale, glint_thres, 255, cv2.THRESH_BINARY)

    # Find blobs in the input image.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    #<!--------------------------------------------------------------------------->
    #<!--                            YOUR CODE HERE                             -->
    #<!--------------------------------------------------------------------------->

    # Analyze each detected blob.
    for contour in contours:

        # Get the contour properties.
        properties = IAMLTools.getContourProperties(contour, ["Area", "Centroid"])

        # Estimate the center of the blob.
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
        else:
            ellipse = cv2.minAreaRect(contour)

        # Get the blob parameters.
        center = properties["Centroid"]

        # Fill out the blob parameters vector.
        centers.append(ellipse[0])

    #<!--------------------------------------------------------------------------->
    #<!--                                                                       -->
    #<!--------------------------------------------------------------------------->

    for uid, coordinate in enumerate(coordinates):
        if (coordinate[0] != -1 and coordinate[1] != -1):
            sdist = 1000000
            sid = -1
            for gid, center in enumerate(centers):
                dist = np.linalg.norm(np.array(coordinate) - np.array(center))
                if (dist < sdist):
                    sdist = dist
                    sid = gid
            bestGlintsIDS[uid] = sid

    # Return the final result.
    centers.append((-1, -1))
    return centers, bestGlintsIDS

# Global variables.
global frame, threshold, ellipses, centers, bestPupilID, json, current_id, \
    coordinates, is_first, target_id, folder, glint_thres

# Set the current dataset folder.
folder = "0138"
target_id = "31"

# Update the following CSV file.
csv_file = "./datasets/%s_dataset.csv" % folder
csv_data = pd.read_csv(csv_file)

# Get all the paths to the eye-video and timestamp files
mov_files = glob.glob("../%s/%s*.mov" % (folder, target_id))
mov_files.sort()

# Kernel size for gaussian blur
kernel_size = 9

# Pupil and target information variables
pupil_x          = []
pupil_y          = []
ellipse_min_axis = []
ellipse_max_axis = []
ellipse_angle    = []
glint_1_x        = []
glint_1_y        = []
glint_2_x        = []
glint_2_y        = []
glint_3_x        = []
glint_3_y        = []
glint_4_x        = []
glint_4_y        = []

# Define the trackbars.
trackbarsValues = {}
trackbarsValues["threshold"] = 60
trackbarsValues["minimum"]   = 25
trackbarsValues["maximum"]   = 50
trackbarsValues["glint"]     = 150

# Create an OpenCV window and some trackbars.
cv2.namedWindow("Threshold", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("threshold", "Threshold", 60, 255, onValuesChange)
cv2.createTrackbar("minimum",   "Threshold", 25, 49, onValuesChange)
cv2.createTrackbar("maximum",   "Threshold", 50, 75, onValuesChange)
cv2.createTrackbar("glint",   "Threshold", 150, 255, onValuesChange)
cv2.imshow("Threshold", np.zeros((1, 640), np.uint8))

cv2.namedWindow("Eye", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Eye", click_and_select)

# Process all the files one by one
for mov_file in mov_files:

    print("Processing movie file: {}...".format(mov_file))

    # Split file path into directory and filename
    p = os.path.split(mov_file)

    # Create a capture video object.
    filename = mov_file
    capture = cv2.VideoCapture(filename)
    
    # Pause in the first frame
    is_first = True
    current_id = None
    coordinates = [(-1, -1)] * 4
    
    # This repetion will run while there is a new frame in the video file or
    # while the user do not press the "q" (quit) keyboard button.
    while True:
        # Capture frame-by-frame.
        retval, frame = capture.read()
    
        # Check if there is a valid frame.
        if not retval:
            # Restart the video.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break

        # Get the detection parameters values.
        threshold = trackbarsValues["threshold"]
        minimum   = trackbarsValues["minimum"]
        maximum   = trackbarsValues["maximum"]
        glint_thres = trackbarsValues["glint"]


        # Pupil detection.
        ellipses, centers, bestPupilID = detectPupil(
            frame, threshold, minimum, maximum, kernel_size)

        # Glints detection
        glints, bestGlintsID = detectGlints(frame, glint_thres, kernel_size)

        # Show the detected pupils.
        showDetectedPupil(frame, threshold, ellipses, centers, bestPupilID,
                          glints, bestGlintsID, current_id, glint_thres)
    
        # Display the captured frame.
        if not is_first:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            while True:
                key = cv2.waitKey(0)
                if (key != ord("1") and key != ord("2") and
                    key != ord("3") and key != ord("4")):
                    break
                else:
                    # Update the current glint ID.
                    current_id = int(chr(key))

                    # Show the detected pupils.
                    showDetectedPupil(frame, threshold, ellipses, centers,
                                      bestPupilID, glints, bestGlintsID,
                                      current_id, glint_thres)

            is_first = False

            # Get the detection parameters values.
            threshold = trackbarsValues["threshold"]
            minimum   = trackbarsValues["minimum"]
            maximum   = trackbarsValues["maximum"]
            glint_thres = trackbarsValues["glint"]

            # Pupil detection.
            ellipses, centers, bestPupilID = detectPupil(
                frame, threshold, minimum, maximum, kernel_size)

            # Glints detection
            glints, bestGlintsID = detectGlints(frame, glint_thres, kernel_size)

            # Show the detected pupils.
            showDetectedPupil(frame, threshold, ellipses, centers, bestPupilID,
                              glints, bestGlintsID, current_id, glint_thres)
    
        # Populate the column vectors
        if (bestPupilID == -1):
            pupil_x         .append(-1)
            pupil_y         .append(-1)
            ellipse_min_axis.append(-1)
            ellipse_max_axis.append(-1)
            ellipse_angle   .append(-1)
            glint_1_x       .append(-1)
            glint_1_y       .append(-1)
            glint_2_x       .append(-1)
            glint_2_y       .append(-1)
            glint_3_x       .append(-1)
            glint_3_y       .append(-1)
            glint_4_x       .append(-1)
            glint_4_y       .append(-1)
        else:
            pupil_x         .append(ellipses[bestPupilID][0][0])
            pupil_y         .append(ellipses[bestPupilID][0][1])
            ellipse_min_axis.append(ellipses[bestPupilID][1][0])
            ellipse_max_axis.append(ellipses[bestPupilID][1][1])
            ellipse_angle   .append(ellipses[bestPupilID][2])
            glint_1_x       .append(glints[bestGlintsID[0]][0])
            glint_1_y       .append(glints[bestGlintsID[0]][1])
            glint_2_x       .append(glints[bestGlintsID[1]][0])
            glint_2_y       .append(glints[bestGlintsID[1]][1])
            glint_3_x       .append(glints[bestGlintsID[2]][0])
            glint_3_y       .append(glints[bestGlintsID[2]][1])
            glint_4_x       .append(glints[bestGlintsID[3]][0])
            glint_4_y       .append(glints[bestGlintsID[3]][1])

    # Update the CSV data.
    indexes = csv_data[csv_data["target_no"] == int(target_id)]["frame_no"].values
    for row, index in enumerate(indexes):
        csv_data.loc[index - 1, "pupil_x"] = pupil_x[row]
        csv_data.loc[index - 1, "pupil_y"] = pupil_y[row]
        csv_data.loc[index - 1, "ellipse_min_axis"] = ellipse_min_axis[row]
        csv_data.loc[index - 1, "ellipse_max_axis"] = ellipse_max_axis[row]
        csv_data.loc[index - 1, "ellipse_angle"] = ellipse_angle[row]
        csv_data.loc[index - 1, "glint_1_x"] = glint_1_x[row]
        csv_data.loc[index - 1, "glint_1_y"] = glint_1_y[row]
        csv_data.loc[index - 1, "glint_2_x"] = glint_2_x[row]
        csv_data.loc[index - 1, "glint_2_y"] = glint_2_y[row]
        csv_data.loc[index - 1, "glint_3_x"] = glint_3_x[row]
        csv_data.loc[index - 1, "glint_3_y"] = glint_3_y[row]
        csv_data.loc[index - 1, "glint_4_x"] = glint_4_x[row]
        csv_data.loc[index - 1, "glint_4_y"] = glint_4_y[row]

    # Write dataframe to the csv file
    csv_data.to_csv(csv_file, index=None, header=True)
    print("Done...")

    capture.release()

# When everything done, release the capture object.
cv2.destroyAllWindows()
