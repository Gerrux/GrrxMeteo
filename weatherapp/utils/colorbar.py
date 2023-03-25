import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(1, 6))
fig.subplots_adjust(right=0.4)


# # Colorbar температуры
# bounds = np.linspace(-40, 50, 10)
# # norm = mpl.colors.BoundaryNorm(bounds, my_cmap.N)
# norm = mpl.colors.Normalize(vmin=-40, vmax=50)
# fig.colorbar(
#     mpl.cm.ScalarMappable(cmap='gist_rainbow_r', norm=norm),
#     cax=ax,
#     extendfrac='auto',
#     ticks=bounds,
#     spacing='uniform',
#     orientation='vertical'
# )
#
# plt.savefig('colorbar.png')

# # Colorbar ветер
# bounds = np.linspace(0, 30, 7)
# # norm = mpl.colors.BoundaryNorm(bounds, my_cmap.N)
# norm = mpl.colors.Normalize(vmin=0, vmax=30)
# fig.colorbar(
#     mpl.cm.ScalarMappable(cmap='rainbow', norm=norm),
#     cax=ax,
#     extendfrac='auto',
#     ticks=bounds,
#     spacing='uniform',
#     orientation='horizontal'
# )
#
# plt.savefig('colorbar.png')

# # Colorbar температуры
# bounds = np.linspace(0, 100, 11)
# # norm = mpl.colors.BoundaryNorm(bounds, my_cmap.N)
# norm = mpl.colors.Normalize(vmin=0, vmax=100)
# fig.colorbar(
#     mpl.cm.ScalarMappable(cmap='gist_gray', norm=norm),
#     cax=ax,
#     extendfrac='auto',
#     ticks=bounds,
#     spacing='uniform',
#     orientation='vertical'
# )
#
# plt.savefig('colorbar.png')


class MidpointNormalize(mpl.colors.Normalize):
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


colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = mpl.colors.LinearSegmentedColormap.from_list("terrain_map", all_colors)
# Отображаем тепловую карту
midnorm = MidpointNormalize(vmin=-500.0, vcenter=0, vmax=4000)

norm = mpl.colors.Normalize(vmin=0, vmax=100)
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=terrain_map, norm=midnorm),
    norm=midnorm,
    cax=ax,
    ticks=[-500, 0, 500, 1000, 2000, 3000, 4000],
    spacing="uniform",
    orientation="vertical",
    extend="both",
)

plt.savefig("colorbar.png")
