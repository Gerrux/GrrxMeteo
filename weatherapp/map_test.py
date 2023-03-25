import io
from statistics import mean

from PIL import Image
from matplotlib import pyplot as plt
import folium

nw_latitude = 67
nw_longitude = 35
se_latitude = 61
se_longitude = 51

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(1, 1, 1)

map = folium.Map(
    location=[mean([se_latitude, nw_latitude]), mean([nw_longitude, se_longitude])],
    zoom_start=6,
    tiles="OpenStreetMap",
    control_scale=True,
    zoom_control=False,
    min_lat=se_latitude,
    max_lat=nw_latitude,
    min_lon=nw_longitude,
    max_lon=se_longitude,
    height=768,
    width=1366,
)

folium.Marker([se_latitude, nw_longitude], popup="<i>Mt. Hood Meadows</i>").add_to(map)
folium.Marker([nw_latitude, se_longitude], popup="<i>Mt. Hood Meadows</i>").add_to(map)
img_data = map._to_png(5)
img = Image.open(io.BytesIO(img_data))
img.save("map_image.png")

ax.imshow(plt.imread("map_image.png"), alpha=1)

# Отображаем карту
plt.tight_layout()
plt.show()

# from math import cos, pi
#
# def calculate_boundaries(lat, lng, zoom, width, height): # -> tuple:
#     upper_left, lower_right = {}, {}
#     C = 40075 # km - Equator distance around the world
#     y = pi * lat / 180 # convert latitude degree to radian
#     S = C * cos(y) / 2 ** (zoom + 8) # km distance of 1 px - https://wiki.openstreetmap.org/wiki/Pt:Zoom_levels
#     S_deg = S * cos(y) / 100 # convert km (distance of 1 px) to degrees (coordinates)
#
#     upper_left['lat'] = lat + height / 2 * S_deg
#     upper_left['lng'] = lng - width / 2 * S_deg
#
#     lower_right['lat'] = lat - height / 2 * S_deg
#     lower_right['lng'] = lng + width / 2 * S_deg
#
#     return upper_left, lower_right
#
# # main
# lat = 50.0
# lng = 50.0
# zoom = 6 # zoom
# width = 80 # considered as pixels
# height = 24
# upper_left, lower_right = calculate_boundaries(lat, lng, zoom, width, height)

# print(upper_left, lower_right)

# import math
# def deg2num(lat_deg, lon_deg, zoom):
#   lat_rad = math.radians(lat_deg)
#   n = 2.0 ** zoom
#   xtile = int((lon_deg + 180.0) / 360.0 * n)
#   ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
#   return (xtile, ytile)
#
# def num2deg(xtile, ytile, zoom):
#   n = 2.0 ** zoom
#   lon_deg = xtile / n * 360.0 - 180.0
#   lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
#   lat_deg = math.degrees(lat_rad)
#   return (lat_deg, lon_deg)
#
# xtile, ytile = deg2num(lat, lng, zoom)
#
# print(num2deg(xtile, ytile, zoom))
