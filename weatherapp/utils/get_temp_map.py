import ssl
from math import sin, cos, pi

from fake_useragent import UserAgent
from random import random
import seaborn as sns
import folium
import folium.plugins
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.image import imread
import aiohttp
import asyncio
import pandas as pd
import numpy as np
import matplotlib
from scipy.interpolate import griddata


# nw_latitude = 81.50
# nw_longitude = 35.15
# se_latitude = 60.35
# se_longitude = 65.55
from weatherapp.utils.get_path import get_path_to_file_from_root


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)


ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

ua = UserAgent()

hdr = {'User-Agent': str(ua.chrome),
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}

nw_latitude = 67
nw_longitude = 35
se_latitude = 61
se_longitude = 51
step = 0.2

latitudes = np.arange(se_latitude, nw_latitude, step).tolist()
longitudes = np.arange(nw_longitude, se_longitude, step).tolist()
urls = []


for latitude in latitudes:
    for longitude in longitudes:
        urls.append(
            f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,winddirection_10m,cloudcover')

# print(len(urls))


async def fetch_coordinates(session, url):
    await asyncio.sleep(random())
    async with session.get(url) as response:
        data = await response.json()
        return data


async def main(urls):
    async with aiohttp.TCPConnector(ssl=ssl_ctx) as connector:
        async with aiohttp.ClientSession(connector=connector, headers=hdr) as session:
            tasks = []
            for url in urls:
                task = asyncio.create_task(fetch_coordinates(session, url))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            return results


def save_ax(ax, filename, **kwargs):
    ax.set_aspect((nw_longitude-se_longitude)/(se_latitude-nw_latitude))
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    plt.savefig(filename, dpi=400, bbox_inches=bbox, transparent=True, **kwargs)
    ax.axis("on")
    im = plt.imread(filename)
    return im


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # jsons = loop.run_until_complete(main(urls))
    # forecast = []
    # for json in jsons:
    #     latitude, longitude = json['latitude'], json['longitude']
    #     forecast_df = pd.DataFrame({'time': json['hourly']['time'],
    #                                 'elevation': json['elevation'],
    #                                 'temp': json['hourly']['temperature_2m'],
    #                                 'humidity': json['hourly']['relativehumidity_2m'],
    #                                 'cloudcover': json['hourly']['cloudcover'],
    #                                 'windspeed': json['hourly']['windspeed_10m'],
    #                                 'winddirection': json['hourly']['winddirection_10m'],
    #                                 })
    #     forecast_df['time'] = pd.to_datetime(forecast_df['time'])
    #     time_now = pd.to_datetime(datetime.now().strftime('%Y/%m/%d %H'))
    #     weather_now = forecast_df.loc[forecast_df['time'] == time_now]
    #     forecast.append({
    #         'latitude': latitude,
    #         'longitude': longitude,
    #         'elevation': float(weather_now['elevation']),
    #         'temperature': float(weather_now['temp']),
    #         'humidity': float(weather_now['humidity']),
    #         'cloudcover': int(weather_now['cloudcover']),
    #         'windspeed': float(weather_now['windspeed']),
    #         'winddirection': int(weather_now['winddirection']),
    #     })
    # forecast_df = pd.DataFrame(forecast)
    # forecast_df.to_csv('data_weather.csv')

    print("Сбор данных окончен")
    forecast_df = pd.read_csv('data_weather.csv')
    time_now = pd.to_datetime(datetime.now().strftime('%Y/%m/%d %H')).strftime('%Y-%m-%d_%H-%M-%S')
    print(time_now)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    lon_min, lon_max = forecast_df['longitude'].min(), forecast_df['longitude'].max()
    lat_min, lat_max = forecast_df['latitude'].min(), forecast_df['latitude'].max()

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Интерполируем значения температуры
    xi = np.linspace(lon_min, lon_max, 1000)
    yi = np.linspace(lat_min, lat_max, 2000)


    # Создание карты температуры
    zi = griddata((forecast_df['longitude'], forecast_df['latitude']), forecast_df['temperature'], (xi[None, :], yi[:, None]),
                  method='linear')

    # Отображаем тепловую карту
    ax.imshow(zi, interpolation='spline16', cmap='gist_rainbow_r',
                   extent=[min(xi), max(xi), min(yi), max(yi)],
                   origin='lower', alpha=1, vmin=-40, vmax=40)

    heat_map = save_ax(ax, get_path_to_file_from_root(f"./maps/{time_now}heat_map.png"))
    plt.cla()

    # Создание карты влажности
    zi = griddata((forecast_df['longitude'], forecast_df['latitude']), forecast_df['humidity'], (xi[None, :], yi[:, None]),
                  method='linear')

    # Отображаем тепловую карту
    ax.imshow(zi, interpolation='spline16', cmap='Blues',
                   extent=[min(xi), max(xi), min(yi), max(yi)],
                   origin='lower', alpha=1, vmin=0, vmax=100)

    humidity_map = save_ax(ax, get_path_to_file_from_root(f"./maps/{time_now}humidity_map.png"))
    plt.cla()

    # Создание карты облачности
    zi = griddata((forecast_df['longitude'], forecast_df['latitude']), forecast_df['cloudcover'], (xi[None, :], yi[:, None]),
                  method='linear')

    # Отображаем тепловую карту
    ax.imshow(zi, interpolation='spline16', cmap='gist_gray',
                   extent=[min(xi), max(xi), min(yi), max(yi)],
                   origin='lower', alpha=1, vmin=0, vmax=100)

    cloudcover_map = save_ax(ax, get_path_to_file_from_root(f"./maps/{time_now}cloudcover_map.png"))
    plt.cla()

    # Создание карты высоты поверхности
    zi = griddata((forecast_df['longitude'], forecast_df['latitude']), forecast_df['elevation'],
                  (xi[None, :], yi[:, None]),
                  method='linear')

    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map',
                                                           all_colors)
    # Отображаем тепловую карту
    midnorm = MidpointNormalize(vmin=-500., vcenter=0.1, vmax=4000)
    ax.imshow(zi, norm=midnorm, cmap=terrain_map,
              extent=[min(xi), max(xi), min(yi), max(yi)],
              origin='lower', alpha=1)

    elevation_map = save_ax(ax, get_path_to_file_from_root(f"./maps/{time_now}elevation_map.png"))
    plt.cla()

    # Создание карты скорости ветра и направления
    zi = griddata((forecast_df['longitude'], forecast_df['latitude']), forecast_df['windspeed'],
                  (xi[None, :], yi[:, None]),
                  method='linear')
    # Отображаем тепловую карту
    ax.imshow(zi, interpolation='spline16', cmap='rainbow',
              extent=[min(xi), max(xi), min(yi), max(yi)],
              origin='lower', alpha=1, vmin=0, vmax=30)
    for dot in forecast_df.itertuples():
        if dot.Index % 2 == 0:
            continue
        u = -0.2 * sin(dot.winddirection*pi/180)
        v = -0.2 * cos(dot.winddirection*pi/180)
        plt.arrow(dot.longitude, dot.latitude, u, v,  length_includes_head=True, head_width=0.04, head_length=0.07, facecolor='black', edgecolor='none')

    wind_map = save_ax(ax, get_path_to_file_from_root(f"./maps/{time_now}wind_map.png"))
    plt.clf()

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