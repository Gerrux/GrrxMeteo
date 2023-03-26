from datetime import datetime

import folium
from django.http import HttpResponse
from django.shortcuts import render
import statistics

# Create your views here.
from weatherapp.models import Map


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
    time_now = datetime.now().strftime("%Y-%m-%d %H")
    maps = Map.objects.filter(timestamp=time_now)
    folium.raster_layers.ImageOverlay(
        image=maps[0].map_path,
        name='<span style="color: red;">Temperature</span>',
        opacity=0.5,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        interactive=True,
        zindex=1,
    ).add_to(map)

    folium.raster_layers.ImageOverlay(
        image=maps[1].map_path,
        name='<span style="color: #5d76cb;">Humidity</span>',
        opacity=0.8,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        interactive=True,
        show=False,
        zindex=1,
    ).add_to(map)

    folium.raster_layers.ImageOverlay(
        image=maps[2].map_path,
        name='<span style="color: grey;">Cloudcover</span>',
        opacity=0.8,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        interactive=True,
        show=False,
        zindex=1,
    ).add_to(map)

    # folium.raster_layers.ImageOverlay(
    #     image="./weatherapp/maps/elevation_map.png",
    #     name='<span style="color: orange;">Elevation</span>',
    #     opacity=0.8,
    #     bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    #     interactive=True,
    #     show=False,
    #     zindex=1,
    # ).add_to(map)

    folium.raster_layers.ImageOverlay(
        image=maps[3].map_path,
        name='<span style="color: darkblue;">Wind</span>',
        opacity=0.6,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        interactive=True,
        show=False,
        zindex=1,
    ).add_to(map)

    # # Добавить colorbar на экран
    # temperature_layer.add_child(folium.plugins.FloatImage('https://i.imgur.com/AI8EZBL.png',
    #                                                       bottom=15, left=93, height='60%'))
    # temperature_layer.add_to(map)

    folium.LayerControl().add_to(map)
    map.render()
    map_html = map._repr_html_()
    return render(request, "weatherapp/index.html", context={"weather_map": map_html})
