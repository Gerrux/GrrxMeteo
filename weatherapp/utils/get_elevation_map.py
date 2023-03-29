from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from weatherapp.utils.get_map import save_ax
from weatherapp.utils.get_path import get_path_to_file_from_root


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.0]
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)

forecast_df = pd.read_csv("data_weather.csv")
forecast_df["time"] = pd.to_datetime(forecast_df["time"])
time_now = pd.to_datetime(datetime.now().strftime("%Y/%m/%d %H"))
time_filename = time_now.strftime("%Y%m%d_%H%M%S")

forecast_now = forecast_df[forecast_df["time"] == time_now]
lon_min, lon_max = forecast_now["longitude"].min(), forecast_now["longitude"].max()
lat_min, lat_max = forecast_now["latitude"].min(), forecast_now["latitude"].max()

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Интерполируем значения температуры
xi = np.linspace(lon_min, lon_max, 1000)
yi = np.linspace(lat_min, lat_max, 2000)


# Создание карты высоты поверхности
zi = griddata(
    (forecast_now["longitude"], forecast_now["latitude"]),
    forecast_now["elevation"],
    (xi[None, :], yi[:, None]),
    method="linear",
)

colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list(
    "terrain_map", all_colors
)
# Отображаем тепловую карту
midnorm = MidpointNormalize(vmin=-500.0, vcenter=0.1, vmax=4000)
ax.imshow(
    zi,
    norm=midnorm,
    cmap=terrain_map,
    extent=[min(xi), max(xi), min(yi), max(yi)],
    origin="lower",
    alpha=1,
)

elevation_map = save_ax(
    ax, get_path_to_file_from_root(f"../media/maps/{time_filename}elevation_map.png")
)
plt.cla()
