import ssl
import time
import folium
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.image import imread
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# async def main():
#     nw_latitude = 81.50
#     nw_longitude = 35.15
#     async with aiohttp.ClientSession() as session:
#         async with session.get(f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,winddirection_10m') as response:
#             answer = await response.json()
#             forecast_df = pd.DataFrame({'time': answer['hourly']['time'], 'temp': answer['hourly']['temperature_2m']})
#             forecast_df['time'] = pd.to_datetime(forecast_df['time'])
#             time_now = pd.to_datetime(datetime.now().strftime('%Y/%m/%d %H'))
#             print(forecast_df.loc[forecast_df['time'] == time_now]['temp'])
#             print(forecast_df.head())

nw_latitude = 81.50
nw_longitude = 35.15
se_latitude = 60.35
se_longitude = 65.55
step = 2

latitudes = np.arange(se_latitude, nw_latitude, step).tolist()
longitudes = np.arange(nw_longitude, se_longitude, step).tolist()
url_list = []

for latitude in latitudes:
    for longitude in longitudes:
        url_list.append(
            f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,winddirection_10m')

print(len(url_list))


async def fetch(session, url):
    async with session.get(url, ssl=ssl.SSLContext()) as response:
        time.sleep(0.05)
        return await response.json()


async def fetch_all(urls, loop):
    async with aiohttp.ClientSession(loop=loop) as session:
        results = await asyncio.gather(*[fetch(session, url) for url in urls], return_exceptions=True)
        return results


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    urls = url_list
    jsons = loop.run_until_complete(fetch_all(urls, loop))
    answer = []
    for json in jsons:
        latitude, longitude = json['latitude'], json['longitude']
        forecast_df = pd.DataFrame({'time': json['hourly']['time'], 'temp': json['hourly']['temperature_2m']})
        forecast_df['time'] = pd.to_datetime(forecast_df['time'])
        time_now = pd.to_datetime(datetime.now().strftime('%Y/%m/%d %H'))
        answer.append({
            'latitude': latitude,
            'longitude': longitude,
            'temperature': float(forecast_df.loc[forecast_df['time'] == time_now]['temp'])
        })
    answer_df = pd.DataFrame(answer)
    print("Сбор данных окончен")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    lon_min, lon_max = answer_df['longitude'].min(), answer_df['longitude'].max()
    lat_min, lat_max = answer_df['latitude'].min(), answer_df['latitude'].max()

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # Интерполируем значения температуры
    xi = np.linspace(lon_min, lon_max, 1000)
    yi = np.linspace(lat_min, lat_max, 1000)
    zi = griddata((answer_df['longitude'], answer_df['latitude']), answer_df['temperature'], (xi[None, :], yi[:, None]),
                  method='linear')

    # Добавление карты на фон
    # Создание карты OpenStreetMap
    map = folium.Map(location=[answer_df['latitude'].mean(), answer_df['longitude'].mean()], zoom_start=6,
                     tiles='OpenStreetMap')

    # Преобразование карты в изображение

    import io
    from PIL import Image

    img_data = map._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img.save('map_image.png')

    ax.imshow(plt.imread('map_image.png'), extent=[lon_min, lon_max, lat_min, lat_max], alpha=1)

    # Отображаем тепловую карту
    im = ax.imshow(zi, interpolation='gaussian', cmap='coolwarm', extent=(min(xi), max(xi), min(yi), max(yi)),
                   origin='lower', alpha=0.5)

    # Добавляем легенду
    cbar = fig.colorbar(im)

    # Отображаем карту
    plt.show()
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
