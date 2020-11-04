import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "../../01_simulated/02_data_analysis/01_homography/00_original/04640_X+000_Y+200_Z+000_dataset_error.csv"
#file = "../../01_simulated/02_data_analysis/01_homography/02_distortion/00000_X-200_Y+050_Z+400_dataset_error.csv"

data = pd.read_csv(file)

x = np.linspace(-200, 200, 21)
y = data.groupby("target_x").error_deg_xyz.apply(lambda c: c.abs().mean()).values
yerr_upper = data.groupby("target_x").error_deg_xyz.apply(lambda c: c.abs().mean()).values
yerr_lower=np.zeros(len(yerr_upper))

plt.bar(x, y,
        width=15,
        color='0.5',
        edgecolor='k',
        yerr=[yerr_lower,yerr_upper],
        capsize=5)
plt.show()
