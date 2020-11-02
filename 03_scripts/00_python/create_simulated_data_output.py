import glob
import numpy as np
import pandas as pd

inputs = [ "../../01_simulated/02_data_analysis/00_interpolate/00_original/*.csv",
           "../../01_simulated/02_data_analysis/00_interpolate/01_camera/*.csv",
           "../../01_simulated/02_data_analysis/00_interpolate/02_distortion/*.csv",
           "../../01_simulated/02_data_analysis/01_homography/00_original/*.csv",
           "../../01_simulated/02_data_analysis/01_homography/01_camera/*.csv",
           "../../01_simulated/02_data_analysis/01_homography/02_distortion/*.csv" ]
outputs = [ "../../01_simulated/03_final_results/00_interpolate_00_original.csv",
            "../../01_simulated/03_final_results/00_interpolate_01_camera.csv",
            "../../01_simulated/03_final_results/00_interpolate_02_distortion.csv",
            "../../01_simulated/03_final_results/01_homography_00_original.csv",
            "../../01_simulated/03_final_results/01_homography_01_camera.csv",
            "../../01_simulated/03_final_results/01_homography_02_distortion.csv" ]
stats = [ "../../01_simulated/03_final_results/00_interpolate_00_original_sts.csv",
          "../../01_simulated/03_final_results/00_interpolate_01_camera_sts.csv",
          "../../01_simulated/03_final_results/00_interpolate_02_distortion_sts.csv",
          "../../01_simulated/03_final_results/01_homography_00_original_sts.csv",
          "../../01_simulated/03_final_results/01_homography_01_camera_sts.csv",
          "../../01_simulated/03_final_results/01_homography_02_distortion_sts.csv" ]

for input, output, stat in zip(inputs, outputs, stats):

    files = glob.glob(input)
    files.sort()

    errors = np.zeros((len(files), 5))

    for i, file in enumerate(files):
        print("%s: %d" % (input, i))
        data = pd.read_csv(file)
        errors[i, 0] = i
        errors[i, 1] = abs(data["error_deg_x"].values).mean()
        errors[i, 2] = abs(data["error_deg_y"].values).mean()
        errors[i, 3] = abs(data["error_deg_z"].values).mean()
        errors[i, 4] = data["error_deg_xyz"].values.mean()

    errors = errors[~np.all(errors[:, 1:4] == 0, axis=1)]

    sts = np.array([[
        errors[:, 1].mean(),
        errors[:, 1].std(),
        errors[:, 2].mean(),
        errors[:, 2].std(),
        errors[:, 3].mean(),
        errors[:, 3].std(),
        errors[:, 4].mean(),
        errors[:, 4].std()
    ]])

    mean = errors[:, 1:3].mean(axis=0)
    std = (sts[0, 1] + sts[0, 3]) / 2
    errors = errors[np.linalg.norm(errors[:, 1:3] - mean, axis=1) < std * 3]

    df = pd.DataFrame(errors, columns=["experiment", "error_deg_x", "error_deg_y", "error_deg_z", "error_deg_xyz"])
    df.experiment = df.experiment.astype(int)
    df.to_csv(output, sep=",", index=False)

    df = pd.DataFrame(sts, columns=["error_deg_x_mean", "error_deg_x_std",
                                    "error_deg_y_mean", "error_deg_y_std",
                                    "error_deg_z_mean", "error_deg_z_std",
                                    "error_deg_xyz_mean", "error_deg_xyz_std"])
    df.to_csv(stat, sep=",", index=False)
