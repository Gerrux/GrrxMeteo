import os
import time as t
from datetime import datetime, timedelta
from math import sin, cos, pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.core.wsgi import get_wsgi_application
from scipy.interpolate import griddata

from weatherapp.utils.get_path import get_path_to_file_from_root
from weatherapp.utils.map_config import *

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grrxmeteoOM.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
application = get_wsgi_application()
from weatherapp.models import Map


def save_ax(ax, filename, **kwargs):
    ax.set_aspect((NW_LONGITUDE - SE_LONGITUDE) / (SE_LATITUDE - NW_LATITUDE))
    ax.set_xlim(NW_LONGITUDE, SE_LONGITUDE)
    ax.set_ylim(NW_LATITUDE, SE_LATITUDE)
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    plt.savefig(filename, dpi=400, bbox_inches=bbox, transparent=True, **kwargs)
    ax.axis("on")
    im = plt.imread(filename)
    return im


if __name__ == "__main__":
    forecast_df = pd.read_csv("data_weather.csv")
    forecast_df["time"] = pd.to_datetime(forecast_df["time"])
    time_now = pd.to_datetime(datetime.now().strftime("%Y/%m/%d %H"))
    maps = []
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    for hour in range(23):
        if hour % 3 != 0:
            continue
        time = time_now + timedelta(hours=hour)
        forecast_now = forecast_df[forecast_df["time"] == time]
        time_filename = time.strftime("%Y%m%d_%H%M%S")
        lon_min, lon_max = (
            forecast_now["longitude"].min(),
            forecast_now["longitude"].max(),
        )
        lat_min, lat_max = (
            forecast_now["latitude"].min(),
            forecast_now["latitude"].max(),
        )

        # Интерполируем значения
        xi = np.linspace(lon_min, lon_max, int(lon_max - lon_min) * 100)
        yi = np.linspace(lat_min, lat_max, int(lat_max - lat_min) * 100)
        # Создание карты температуры
        zi = griddata(
            (forecast_now["longitude"], forecast_now["latitude"]),
            forecast_now["temperature"],
            (xi[None, :], yi[:, None]),
            method="linear",
        )
        # Отображаем тепловую карту
        ax.imshow(
            zi,
            interpolation="gaussian",
            cmap="gist_rainbow_r",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=-40,
            vmax=40,
        )
        filename = get_path_to_file_from_root(
            f"../media/maps/{time_filename}heat_map.png"
        )
        heat_map = save_ax(ax, filename)
        maps.append(("heat_map", time, filename))
        plt.cla()
        # Создание карты влажности
        zi = griddata(
            (forecast_now["longitude"], forecast_now["latitude"]),
            forecast_now["humidity"],
            (xi[None, :], yi[:, None]),
            method="linear",
        )

        # Отображаем тепловую карту
        ax.imshow(
            zi,
            interpolation="gaussian",
            cmap="Blues",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=0,
            vmax=100,
        )
        filename = get_path_to_file_from_root(
            f"../media/maps/{time_filename}humidity_map.png"
        )
        humidity_map = save_ax(ax, filename)
        maps.append(("humidity_map", time, filename))
        plt.cla()

        # Создание карты облачности
        zi = griddata(
            (forecast_now["longitude"], forecast_now["latitude"]),
            forecast_now["cloudcover"],
            (xi[None, :], yi[:, None]),
            method="linear",
        )

        # Отображаем тепловую карту
        ax.imshow(
            zi,
            interpolation="gaussian",
            cmap="gist_gray",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=0,
            vmax=100,
        )
        filename = get_path_to_file_from_root(
            f"../media/maps/{time_filename}cloudcover_map.png"
        )
        cloudcover_map = save_ax(ax, filename)
        maps.append(("cloudcover_map", time, filename))
        plt.cla()

        # Создание карты скорости ветра и направления
        zi = griddata(
            (forecast_now["longitude"], forecast_now["latitude"]),
            forecast_now["windspeed"],
            (xi[None, :], yi[:, None]),
            method="linear",
        )

        # Отображаем тепловую карту
        ax.imshow(
            zi,
            interpolation="gaussian",
            cmap="rainbow",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=0,
            vmax=30,
        )
        i = 0
        # Отображаем стрелки направления ветра
        for dot in forecast_now.itertuples():
            i += 1
            if i % 5 == 0:
                u = -0.2 * sin(dot.winddirection * pi / 180)
                v = -0.2 * cos(dot.winddirection * pi / 180)
                plt.arrow(
                    dot.longitude,
                    dot.latitude,
                    u,
                    v,
                    length_includes_head=True,
                    head_width=0.04,
                    head_length=0.07,
                    facecolor="black",
                    edgecolor="none",
                )
        filename = get_path_to_file_from_root(
            f"../media/maps/{time_filename}wind_map.png"
        )
        wind_map = save_ax(ax, filename)
        maps.append(("wind_map", time, filename))
        plt.cla()
    Map.objects.all().delete()
    Map.objects.bulk_create(
        [Map(**{"title": m[0], "timestamp": m[1], "map_path": m[2]}) for m in maps]
    )
