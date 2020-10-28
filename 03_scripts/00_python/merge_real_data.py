import glob
import pandas as pd

output = "homography_method03.csv"
filenames = glob.glob("../../02_real/02_data_analysis/01_homography/02_distortion/*.csv")
filenames.sort()

non_valid = [ 5, 23, 24, 27, 28, 42, 44, 45, 47, 50, 51, 54, 55, 58,
              62, 73, 75, 82, 83, 97, 98, 115, 119, 124, 125, 134,
              135, 136, 139, 147, 148, 151, 164 ]

data = pd.DataFrame(columns=["experiment", "target_x", "target_y", "gaze_x", "gaze_y", "error_px_x", "error_px_y", "error_px_xy",
                "error_deg_x", "error_deg_y", "error_deg_xy"])

for i, filename in enumerate(filenames):
    if i not in non_valid:
        df = pd.read_csv(filename, sep=",", header=0)
        df.insert(0, "experiment", [i] * 35, True)
        data = data.append(df, ignore_index = True)

data.to_csv(output, sep=",")