import re
import sys

import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    log_file = "log_train"
else:
    log_file = sys.argv[1]

pattern = re.compile(
    r"Updates (\d+), num timesteps (\d+), FPS \d+, mean/median reward (-?\d+\.\d+)/(-?\d+\.\d+), min/max reward (-?\d+\.\d+)/(-?\d+\.\d+), .*$"
)

timestep = []
avg_reward = []
min_reward = []
max_reward = []

with open(log_file, "r") as f:
    for line in f:
        results = pattern.match(line)
        if results is None:
            continue
        timestep.append(int(results[2]))
        avg_reward.append(float(results[3]))
        min_reward.append(float(results[5]))
        max_reward.append(float(results[6]))


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


factor = 10
x = timestep[:-factor]
y = smooth(avg_reward, factor)[:-factor]
y_max = smooth(max_reward, factor)[:-factor]
y_min = smooth(min_reward, factor)[:-factor]

plt.plot(x, y, color="blue", linewidth=1)
plt.plot(x, y_max, color="skyblue", linewidth=1)
plt.plot(x, y_min, color="skyblue", linewidth=1)

plt.fill_between(x, y, y_max, facecolor="lightskyblue", interpolate=True)
plt.fill_between(x, y_min, y, facecolor="lightskyblue", interpolate=True)
plt.show()
