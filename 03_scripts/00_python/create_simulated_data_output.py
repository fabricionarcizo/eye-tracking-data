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

    errors = np.zeros((len(files), 11))

    for i, file in enumerate(files):
        print("%s: %d" % (input, i))
        data = pd.read_csv(file)
        errors[i, 0] = i
        errors[i, 1] = int(file.split("/")[-1].split("_")[1][1:])
        errors[i, 2] = int(file.split("/")[-1].split("_")[2][1:])
        errors[i, 3] = int(file.split("/")[-1].split("_")[3][1:])

        errors[i, 4] = (data["error_px_x"].values).mean()
        errors[i, 5] = (data["error_px_y"].values).mean()
        errors[i, 6] = (data["error_px_xy"].values).mean()

        errors[i,  7] = (data["error_deg_x"].values).mean()
        errors[i,  8] = (data["error_deg_y"].values).mean()
        errors[i,  9] = (data["error_deg_z"].values).mean()
        errors[i, 10] = (data["error_deg_xyz"].values).mean()

    errors = errors[~np.all(errors[:, 1:4] == 0, axis=1)]

    sts = np.array([[
        errors[:,  4].mean(),
        errors[:,  4].std(),
        errors[:,  5].mean(),
        errors[:,  5].std(),
        errors[:,  6].mean(),
        errors[:,  6].std(),
        errors[:,  7].mean(),
        errors[:,  7].std(),
        errors[:,  8].mean(),
        errors[:,  8].std(),
        errors[:,  9].mean(),
        errors[:,  9].std(),
        errors[:, 10].mean(),
        errors[:, 10].std()
    ]])

    mean = errors[:, 7:9].mean(axis=0)
    std = np.array([sts[0, 7], sts[0, 9]])
    errors = errors[(errors[:, 7] - mean[0]) < std[0] * 3]
    errors = errors[(errors[:, 8] - mean[1]) < std[1] * 3]

    df = pd.DataFrame(errors, columns=[ "experiment", "camera_pos_x", "camera_pos_y", "camera_pos_z",
                                        "error_px_x", "error_px_y", "error_px_xy",
                                        "error_deg_x", "error_deg_y", "error_deg_z", "error_deg_xyz" ])
    df.experiment = df.experiment.astype(int)
    df.to_csv(output, sep=",", index=False)

    df = pd.DataFrame(sts, columns=[ "error_px_x_mean", "error_px_x_std",
                                     "error_px_y_mean", "error_px_y_std",
                                     "error_px_xy_mean", "error_px_xy_std",
                                     "error_deg_x_mean", "error_deg_x_std",
                                     "error_deg_y_mean", "error_deg_y_std",
                                     "error_deg_z_mean", "error_deg_z_std",
                                     "error_deg_xyz_mean", "error_deg_xyz_std" ])
    df.to_csv(stat, sep=",", index=False)
