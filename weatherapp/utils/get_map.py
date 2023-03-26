import os
from datetime import datetime, timedelta
from math import sin, cos, pi

import django
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.conf import settings
from scipy.interpolate import griddata


from weatherapp.utils.get_path import get_path_to_file_from_root
from weatherapp.utils.map_config import *
import sys
from django.core.wsgi import get_wsgi_application


# sys.path.extend([get_path_to_file_from_root('')])
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grrxmeteoOM.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
application = get_wsgi_application()
# settings.configure()
from weatherapp.models import Map
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


def save_ax(ax, filename, **kwargs):
    ax.set_aspect((NW_LONGITUDE - SE_LONGITUDE) / (SE_LATITUDE - NW_LATITUDE))
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    plt.savefig(filename, dpi=400, bbox_inches=bbox, transparent=True, **kwargs)
    ax.axis("on")
    im = plt.imread(filename)
    return im


if __name__ == "__main__":
    Map.objects.all().delete()
    forecast_df = pd.read_csv("data_weather.csv")
    forecast_df["time"] = pd.to_datetime(forecast_df["time"])
    time_now = pd.to_datetime(datetime.now().strftime("%Y/%m/%d %H"))
    maps = []
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    for hour in range(12):
        time = time_now + timedelta(hours=hour)
        forecast_now = forecast_df[forecast_df["time"] == time]
        time_filename = time.strftime("%Y%m%d_%H%M%S")
        timestamp = time.strftime("%Y-%m-%d %H:00[:00[.000000]][0]")
        lon_min, lon_max = forecast_now["longitude"].min(), forecast_now["longitude"].max()
        lat_min, lat_max = forecast_now["latitude"].min(), forecast_now["latitude"].max()

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Интерполируем значения температуры
        xi = np.linspace(lon_min, lon_max, 1000)
        yi = np.linspace(lat_min, lat_max, 2000)

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
            interpolation="spline16",
            cmap="gist_rainbow_r",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=-40,
            vmax=40,
        )
        filename = get_path_to_file_from_root(f"../media/maps/{time_filename}heat_map.png")
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
            interpolation="spline16",
            cmap="Blues",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=0,
            vmax=100,
        )
        filename =get_path_to_file_from_root(f"../media/maps/{time_filename}humidity_map.png")
        humidity_map = save_ax(
            ax, filename
        )
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
            interpolation="spline16",
            cmap="gist_gray",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=0,
            vmax=100,
        )
        filename = get_path_to_file_from_root(f"../media/maps/{time_filename}cloudcover_map.png")
        cloudcover_map = save_ax(
            ax, filename
        )
        maps.append(("cloudcover_map", time, filename))
        plt.cla()

        # # Создание карты высоты поверхности
        # zi = griddata(
        #     (forecast_now["longitude"], forecast_now["latitude"]),
        #     forecast_now["elevation"],
        #     (xi[None, :], yi[:, None]),
        #     method="linear",
        # )
        #
        # colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
        # colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
        # all_colors = np.vstack((colors_undersea, colors_land))
        # terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        #     "terrain_map", all_colors
        # )
        # # Отображаем тепловую карту
        # midnorm = MidpointNormalize(vmin=-500.0, vcenter=0.1, vmax=4000)
        # ax.imshow(
        #     zi,
        #     norm=midnorm,
        #     cmap=terrain_map,
        #     extent=[min(xi), max(xi), min(yi), max(yi)],
        #     origin="lower",
        #     alpha=1,
        # )
        #
        # elevation_map = save_ax(
        #     ax, get_path_to_file_from_root(f"../media/maps/{time_filename}elevation_map.png")
        # )
        # plt.cla()

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
            interpolation="spline16",
            cmap="rainbow",
            extent=[min(xi), max(xi), min(yi), max(yi)],
            origin="lower",
            alpha=1,
            vmin=0,
            vmax=30,
        )

        # Отображаем стрелки направления ветра
        for dot in forecast_now.itertuples():
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
        filename = get_path_to_file_from_root(f"../media/maps/{time_filename}wind_map.png")
        wind_map = save_ax(ax, filename)
        maps.append(("wind_map", time, filename))
        plt.cla()
        print(time_filename)
    Map.objects.bulk_create([Map(**{'title': m[0],
                                    'timestamp': m[1],
                                    'map_path': m[2]})
                                    for m in maps])
        # Добавление карты на фон

        # # Создание карты OpenStreetMap
        # map = folium.Map(location=[forecast_df['latitude'].mean(), forecast_df['longitude'].mean()], zoom_start=7, min_zoom=7,
        #                  tiles='OpenStreetMap')
        #
        # folium.raster_layers.ImageOverlay(
        #     image=heat_map,
        #     name='<span style="color: red;">Temperature</span>',
        #     opacity=0.8,
        #     bounds=[[min(yi), min(xi)], [max(yi), max(xi)]],
        #     interactive=True,
        #     zindex=1
        # ).add_to(map)
        #
        # folium.raster_layers.ImageOverlay(
        #     image=humidity_map,
        #     name='<span style="color: #5d76cb;">Humidity</span>',
        #     opacity=0.8,
        #     bounds=[[min(yi), min(xi)], [max(yi), max(xi)]],
        #     interactive=True,
        #     show=False,
        #     zindex=1
        # ).add_to(map)
        #
        # folium.raster_layers.ImageOverlay(
        #     image=cloudcover_map,
        #     name='<span style="color: grey;">Cloudcover</span>',
        #     opacity=0.8,
        #     bounds=[[min(yi), min(xi)], [max(yi), max(xi)]],
        #     interactive=True,
        #     show=False,
        #     zindex=1
        # ).add_to(map)
        #
        # folium.raster_layers.ImageOverlay(
        #     image=elevation_map,
        #     name='<span style="color: orange;">Elevation</span>',
        #     opacity=0.8,
        #     bounds=[[min(yi), min(xi)], [max(yi), max(xi)]],
        #     interactive=True,
        #     show=False,
        #     zindex=1
        # ).add_to(map)
        #
        # folium.raster_layers.ImageOverlay(
        #     image=wind_map,
        #     name='<span style="color: darkblue;">Wind</span>',
        #     opacity=0.6,
        #     bounds=[[min(yi), min(xi)], [max(yi), max(xi)]],
        #     interactive=True,
        #     show=False,
        #     zindex=1
        # ).add_to(map)
        #
        # # temperature_layer.add_child(folium.plugins.FloatImage('https://i.imgur.com/AI8EZBL.png',
        # #                                                       bottom=15, left=93, height='60%'))
        # # temperature_layer.add_to(map)
        #
        # folium.LayerControl().add_to(map)
        #
        # # Преобразование карты в html
        # map.save('map.html')
