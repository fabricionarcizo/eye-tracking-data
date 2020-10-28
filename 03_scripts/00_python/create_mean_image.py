# <!--------------------------------------------------------------------------->
# <!--                   ITU - IT University of Copenhagen                   -->
# <!--                      Computer Science Department                      -->
# <!--                    Eye Information Research Group                     -->
# <!--       Introduction to Image Analysis and Machine Learning Course      -->
# <!-- File       : create_mean_image.py                                     -->
# <!-- Description: Script to calculate and create the mean of 150 frames    -->
# <!--            : available in the eye videos                              -->
# <!-- Author     : Fabricio Batista Narcizo                                 -->
# <!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
# <!--            : narcizo[at]itu[dot]dk                                    -->
# <!-- Responsible: Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)         -->
# <!--              Zaheer Ahmed (zahm[at]itu[dot]dk)                        -->
# <!--              Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
# <!-- Information: No additional information                                -->
# <!-- Date       : 16/10/2018                                               -->
# <!-- Change     : 16/10/2018 - Creation of this script                     -->
# <!-- Review     : 16/10/2019 - Finalized                                   -->
# <!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019101601 $"

################################################################################
import cv2
import glob
import os
import numpy as np

################################################################################

# Process the video from all users.
for i in range(166):

    # Get the folder name.
    folder = "%04d" % i

    # Get all the paths to the eye-videos.
    files = glob.glob("/Users/eyeinfo/Desktop/outputs/%s/*.mov" % folder)
    files.sort()

    # Create the output folder.
    if not os.path.exists("mean/%s" % folder):
        os.makedirs("mean/%s" % folder)

    # Process individually each video.
    for file in files:

        # Open the current video.
        video = cv2.VideoCapture(file)

        # Read the first frame.
        ret, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float)

        # Grab the frames.
        while ret:

            # Get the current frame.
            ret, frame = video.read()
            if ret:

                # Convert to grayscale.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Add the current frame to the accumulator.
                image += frame

        # Calculate the mean.
        image /= 150
        image = image.astype(np.uint8).copy()

        # Save the current image.
        filename = os.path.basename(file).split(".")[0]
        cv2.imwrite("mean/%s/%s.png" % (folder, filename), image)