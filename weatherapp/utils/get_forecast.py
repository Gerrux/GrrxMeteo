import asyncio
import ssl
from datetime import datetime, timedelta
from random import random

import aiohttp
import numpy as np
import pandas as pd
from fake_useragent import UserAgent

from weatherapp.utils.map_config import *

ua = UserAgent()

hdr = {
    "User-Agent": str(ua.chrome),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

urls = []


for latitude in np.arange(SE_LATITUDE, NW_LATITUDE, STEP_DEGREES):
    for longitude in np.arange(NW_LONGITUDE, SE_LONGITUDE, STEP_DEGREES):
        urls.append(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m,winddirection_10m,cloudcover"
        )

# print(len(urls))


async def fetch_coordinate(session, url):
    await asyncio.sleep(random())
    async with session.get(url) as response:
        data = await response.json()
        return data


async def fetch_all_coordinates(urls):
    async with aiohttp.TCPConnector(ssl=ssl_ctx) as connector:
        async with aiohttp.ClientSession(connector=connector, headers=hdr) as session:
            tasks = []
            for url in urls:
                task = asyncio.create_task(fetch_coordinate(session, url))
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            return results


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    jsons = loop.run_until_complete(fetch_all_coordinates(urls))
    forecast = []
    time_now = datetime.now()
    hour_list = [time_now + datetime.timedelta(hours=x) for x in range(24)]
    for json in jsons:
        if pd.to_datetime(json["hourly"]["time"]) not in hour_list:
            continue
        latitude, longitude = json["latitude"], json["longitude"]
        forecast_df = pd.DataFrame(
            {
                "time": json["hourly"]["time"],
                "elevation": json["elevation"],
                "temp": json["hourly"]["temperature_2m"],
                "humidity": json["hourly"]["relativehumidity_2m"],
                "cloudcover": json["hourly"]["cloudcover"],
                "windspeed": json["hourly"]["windspeed_10m"],
                "winddirection": json["hourly"]["winddirection_10m"],
            }
        )
        forecast_df["time"] = pd.to_datetime(forecast_df["time"])
        time_now = pd.to_datetime(datetime.now().strftime("%Y/%m/%d %H"))
        time_after_day = pd.to_datetime((datetime.now() + timedelta(days=1)).strftime("%Y/%m/%d %H"))
        forecast_day_df = forecast_df.loc[(forecast_df["time"] >= time_now) & (forecast_df["time"] <= time_after_day)]
        for hour in range(24):
            time = time_now + timedelta(hours=hour)
            forecast_now = forecast_day_df[forecast_day_df["time"] == time]
            forecast.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "time": time,
                    "elevation": float(forecast_now["elevation"]),
                    "temperature": float(forecast_now["temp"]),
                    "humidity": float(forecast_now["humidity"]),
                    "cloudcover": int(forecast_now["cloudcover"]),
                    "windspeed": float(forecast_now["windspeed"]),
                    "winddirection": int(forecast_now["winddirection"]),
                }
            )
    forecast_df = pd.DataFrame(forecast)
    forecast_df.to_csv("data_weather.csv")
