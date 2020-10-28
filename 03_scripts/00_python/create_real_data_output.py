import glob
import numpy as np
import pandas as pd

non_valid = [ 5, 23, 24, 27, 28, 42, 44, 45, 47, 50, 51, 54, 55, 58,
              62, 73, 75, 82, 83, 97, 98, 115, 119, 124, 125, 134,
              135, 136, 139, 147, 148, 151, 164 ]

inputs = [ "../../02_real/02_data_analysis/00_interpolate/00_original/*.csv",
           "../../02_real/02_data_analysis/00_interpolate/01_camera/*.csv",
           "../../02_real/02_data_analysis/00_interpolate/02_distortion/*.csv",
           "../../02_real/02_data_analysis/01_homography/00_original/*.csv",
           "../../02_real/02_data_analysis/01_homography/01_camera/*.csv",
           "../../02_real/02_data_analysis/01_homography/02_distortion/*.csv" ]
outputs = [ "../../02_real/03_final_results/00_interpolate_00_original.csv",
            "../../02_real/03_final_results/00_interpolate_01_camera.csv",
            "../../02_real/03_final_results/00_interpolate_02_distortion.csv",
            "../../02_real/03_final_results/01_homography_00_original.csv",
            "../../02_real/03_final_results/01_homography_01_camera.csv",
            "../../02_real/03_final_results/01_homography_02_distortion.csv" ]
stats = [ "../../02_real/03_final_results/00_interpolate_00_original_sts.csv",
          "../../02_real/03_final_results/00_interpolate_01_camera_sts.csv",
          "../../02_real/03_final_results/00_interpolate_02_distortion_sts.csv",
          "../../02_real/03_final_results/01_homography_00_original_sts.csv",
          "../../02_real/03_final_results/01_homography_01_camera_sts.csv",
          "../../02_real/03_final_results/01_homography_02_distortion_sts.csv" ]

for input, output, stat in zip(inputs, outputs, stats):

    files = glob.glob(input)
    files.sort()

    errors = np.zeros((len(files), 5))

    for i, file in enumerate(files):
        if i not in non_valid:
            data = pd.read_csv(file)
            errors[i, 0] = i
            errors[i, 1] = i % 2
            errors[i, 2] = abs(data["error_deg_x"].values).mean()
            errors[i, 3] = abs(data["error_deg_y"].values).mean()
            errors[i, 4] = data["error_deg_xy"].values.mean()

    errors = errors[~np.all(errors[:, 1:4] == 0, axis=1)]

    df = pd.DataFrame(errors, columns=["experiment", "eye_id", "error_deg_x", "error_deg_y", "error_deg_xy"])
    df.experiment = df.experiment.astype(int)
    df.eye_id = df.eye_id.astype(int)
    df.to_csv(output, sep=",", index=False)

    sts = np.array([[
        df["error_deg_x"].values.mean(),
        df["error_deg_x"].values.std(),
        df["error_deg_y"].values.mean(),
        df["error_deg_y"].values.std(),
        df["error_deg_xy"].values.mean(),
        df["error_deg_xy"].values.std()
    ]])

    df = pd.DataFrame(sts, columns=["error_deg_x_mean", "error_deg_x_std",
                                    "error_deg_y_mean", "error_deg_y_std",
                                    "error_deg_xy_mean", "error_deg_xy_std"])
    df.to_csv(stat, sep=",", index=False)
