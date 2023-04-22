import asyncio
from datetime import datetime, timedelta
import time as t
import folium
from branca.element import CssLink, JavascriptLink
from django.http import HttpResponse
from django.shortcuts import render
import statistics
# Create your views here.
from weatherapp.models import Map
from weatherapp.utils.get_forecast import get_single_geolocation_forecast
from weatherapp.utils.get_path import get_client_ip, get_location_by_ip


def index(request):
    max_lat = 67
    min_lat = 61
    max_lon = 51
    min_lon = 35
    mean_lat = statistics.mean([min_lat, max_lat])
    mean_lon = statistics.mean([min_lon, max_lon])
    map = folium.Map(
        location=[mean_lat, mean_lon], zoom_start=7, min_zoom=7, tiles="OpenStreetMap"
    )
    time_now = datetime.now()
    list_of_times = []
    for i in range(0, 25, 3):
        list_of_times.append(
            (time_now + timedelta(hours=i)).strftime("%Y-%m-%d %H:00:00")
        )
    maps = Map.objects.filter(timestamp__in=list_of_times)
    hour = 0
    for m in maps:
        buf = hour // 4 * 3
        match m.title:
            case "heat_map":
                folium.raster_layers.ImageOverlay(
                    image=m.map_path,
                    name=f'<span style="color: red;" id="temperature{buf}h">temperature{buf}h</span>',
                    opacity=0.5,
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    interactive=True,
                    show=False,
                    zindex=1,
                ).add_to(map)
            case "humidity_map":
                folium.raster_layers.ImageOverlay(
                    image=m.map_path,
                    name=f'<span style="color: #5d76cb;" id="humidity{buf}h">humidity{buf}h</span>',
                    opacity=0.8,
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    interactive=True,
                    show=False,
                    zindex=1,
                ).add_to(map)
            case "cloudcover_map":
                folium.raster_layers.ImageOverlay(
                    image=m.map_path,
                    name=f'<span style="color: grey;" id="cloudcover{buf}h">cloudcover{buf}h</span>',
                    opacity=0.8,
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    interactive=True,
                    show=False,
                    zindex=1,
                ).add_to(map)
            case "wind_map":
                folium.raster_layers.ImageOverlay(
                    image=m.map_path,
                    name=f'<span style="color: darkblue;" id="wind{buf}h">wind{buf}h</span>',
                    opacity=0.6,
                    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                    interactive=True,
                    show=False,
                    zindex=1,
                ).add_to(map)
        hour += 1
    # folium.raster_layers.ImageOverlay(
    #     image="./weatherapp/maps/elevation_map.png",
    #     name='<span style="color: orange;">Elevation</span>',
    #     opacity=0.8,
    #     bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    #     interactive=True,
    #     show=False,
    #     zindex=1,
    # ).add_to(map)

    # # Добавить colorbar на экран
    # temperature_layer.add_child(folium.plugins.FloatImage('https://i.imgur.com/AI8EZBL.png',
    #                                                       bottom=15, left=93, height='60%'))
    # temperature_layer.add_to(map)
    hour_now = int(time_now.strftime("%H"))
    hours = [hour % 24 for hour in range(hour_now, hour_now + 24, 3)]
    folium.LayerControl().add_to(map)
    map.get_root().html.add_child(
        folium.Element(
            f"""
        <ul class="navbar-map-type">
            <li class="navbar-btn-t active" id="temperature"><span>Температура</span></li>
            <li class="navbar-btn-t" id="humidity"><span>Влажность</span></li>
            <li class="navbar-btn-t" id="cloudcover"><span>Облачность</span></li>
            <li class="navbar-btn-t" id="wind"><span>Ветер</span></li>
            <li class="navbar-btn-t" id="elevation"><span>Рельеф</span></li>
        </ul>
        <ul class="navbar-map-hour">
            <li class="navbar-btn-h active" id="0h"><span>{hours[0]}:00</span></li>
            <li class="navbar-btn-h" id="3h"><span>{hours[1]}:00</span></li>
            <li class="navbar-btn-h" id="6h"><span>{hours[2]}:00</span></li>
            <li class="navbar-btn-h" id="9h"><span>{hours[3]}:00</span></li>
            <li class="navbar-btn-h" id="12h"><span>{hours[4]}:00</span></li>
            <li class="navbar-btn-h" id="15h"><span>{hours[5]}:00</span></li>
            <li class="navbar-btn-h" id="18h"><span>{hours[6]}:00</span></li>
            <li class="navbar-btn-h" id="21h"><span>{hours[7]}:00</span></li>
        </ul>
        
        <img src="https://i.imgur.com/YWh1FIg.png" alt="colorbar" class="colorbar active-colorbar" id="temperature-colorbar">
        <img src="https://i.imgur.com/RAtaa9l.png" alt="colorbar" class="colorbar" id="elevation-colorbar">
        <img src="https://i.imgur.com/9y12BZi.png" alt="colorbar" class="colorbar" id="cloudcover-colorbar">
        <img src="https://i.imgur.com/7FscwVd.png" alt="colorbar" class="colorbar" id="humidity-colorbar">
        <img src="https://i.imgur.com/LzTGVwN.png" alt="colorbar" class="colorbar" id="wind-colorbar">
    """
        )
    )
    map.get_root().header.add_child(CssLink("../static/weatherapp/styles/map.css"))
    map.get_root().html.add_child(JavascriptLink("../static/weatherapp/scripts/map.js"))
    map.render()
    map_html = map._repr_html_()

    ip = get_client_ip(request)
    location = get_location_by_ip(ip)
    forecast_on_ip = asyncio.run(get_single_geolocation_forecast(location))
    return render(
        request,
        "weatherapp/index.html",
        context={
            "weather_map": map_html,
            "now": datetime.strftime(datetime.now(), "%d-%m-%Y %H:%M:%S"),
            "ip": ip,
            "client_temperature": forecast_on_ip[0]["current_weather"]["temperature"]
        },
    )
