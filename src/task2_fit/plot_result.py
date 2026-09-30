import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = os.path.join(base_dir,"result","task2_fit/")


# ==========================
# 读取角度拟合数据
# ==========================

fit = pd.read_csv(
    path + "fit.csv"
)


t = fit["time"]
theta_data = fit["theta_data"]
theta_fit = fit["theta_fit"]
error = fit["error"]



# ==========================
# 1. 角度拟合图
# ==========================

plt.figure(figsize=(10,5))


plt.plot(
    t,
    theta_data,
    label="Measured theta"
)


plt.plot(
    t,
    theta_fit,
    label="Fitted theta"
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Angle (rad)"
)


plt.title(
    "Angular Position Fitting"
)


plt.legend()

plt.grid()


plt.tight_layout()


plt.savefig(
    path+"fit_comparison.png",
    dpi=300
)


plt.close()



# ==========================
# 2. 残差图
# ==========================

plt.figure(figsize=(10,5))


plt.plot(
    t,
    error
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Residual (rad)"
)


plt.title(
    "Angular Residual"
)


plt.grid()


plt.tight_layout()


plt.savefig(
    path+"residuals.png",
    dpi=300
)


plt.close()



# ==========================
# 3. 角速度拟合
# ==========================


omega = pd.read_csv(
    path+"omega_fit.csv"
)


plt.figure(figsize=(10,5))


plt.plot(
    omega["time"],
    omega["omega_data"],
    label="Measured omega"
)


plt.plot(
    omega["time"],
    omega["omega_fit"],
    label="Fitted omega"
)


plt.xlabel(
    "Time (s)"
)


plt.ylabel(
    "Angular velocity (rad/s)"
)


plt.title(
    "Angular Velocity Fitting"
)


plt.legend()


plt.grid()


plt.tight_layout()


plt.savefig(
    path+"angular_velocity.png",
    dpi=300
)


plt.close()



# ==========================
# 输出RMSE
# ==========================


theta_rmse = np.sqrt(
    np.mean(
        error**2
    )
)


omega_error = omega["error"]


omega_rmse = np.sqrt(
    np.mean(
        omega_error**2
    )
)



print("====================")
print(
    "Theta RMSE:",
    theta_rmse,
    "rad"
)


print(
    "Omega RMSE:",
    omega_rmse,
    "rad/s"
)

print("====================")